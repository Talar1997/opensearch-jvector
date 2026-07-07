/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.jvector;

import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.util.Bits;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.search.VectorScorer;

import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.Arrays;

public class JVectorFloatVectorValues extends FloatVectorValues {
    public static final int NO_VECTOR = -1;
    private static final VectorTypeSupport VECTOR_TYPE_SUPPORT = VectorizationProvider.getInstance().getVectorTypeSupport();

    private final OnDiskGraphIndex.View view;
    private final VectorSimilarityFunction similarityFunction;
    private final int dimension;
    private final int size;
    private final GraphNodeIdToDocMap graphNodeIdToDocMap;

    public JVectorFloatVectorValues(
        OnDiskGraphIndex onDiskGraphIndex,
        VectorSimilarityFunction similarityFunction,
        GraphNodeIdToDocMap graphNodeIdToDocMap
    ) throws IOException {
        this.view = onDiskGraphIndex.getView();
        this.dimension = view.dimension();
        this.size = view.size();
        this.similarityFunction = similarityFunction;
        this.graphNodeIdToDocMap = graphNodeIdToDocMap;
    }

    @Override
    public int dimension() {
        return dimension;
    }

    @Override
    public int size() {
        return size;
    }

    // This allows us to access the vector without copying it to float[]
    public VectorFloat<?> vectorFloatValue(int ord) {
        return view.getVector(ord);
    }

    public DocIndexIterator iterator() {
        return new DocIndexIterator() {
            private int docId = -1;
            private final Bits liveNodes = view.liveNodes();

            @Override
            public long cost() {
                return size();
            }

            @Override
            public int index() {
                return graphNodeIdToDocMap.getJVectorNodeId(docId);
            }

            @Override
            public int docID() {
                return docId;
            }

            @Override
            public int nextDoc() throws IOException {
                // Advance to the next node docId starts from -1 which is why we need to increment docId by 1
                // until maxDoc is reached. If the document has vector field but no value (== null), there will
                // gaps in the document <-> node maps, index() will return NO_VECTOR_OR_DELETED_DOC in such cases.
                while (docId < graphNodeIdToDocMap.getMaxDoc() - 1) {
                    docId++;
                    if (liveNodes.get(docId)) {
                        return docId;
                    }
                }
                docId = NO_MORE_DOCS;

                return docId;
            }

            @Override
            public int advance(int target) throws IOException {
                return slowAdvance(target);
            }
        };
    }

    /**
     * Constructs an iterator that iterates over vectors that have corresponding nodes in the graph (skipping the gaps with non-live / NO_VECTORS nodes).
     * @return an iterator that iterates over vectors that have corresponding nodes in the graph (skipping the gaps with non-live / NO_VECTORS nodes)
     */
    public DocIndexIterator vectorIterator() {
        return new DocIndexIterator() {
            private int docId = -1;
            private final Bits liveNodes = view.liveNodes();

            @Override
            public long cost() {
                return size();
            }

            @Override
            public int index() {
                return graphNodeIdToDocMap.getJVectorNodeId(docId);
            }

            @Override
            public int docID() {
                return docId;
            }

            @Override
            public int nextDoc() throws IOException {
                // Advance to the next node docId starts from -1 which is why we need to increment docId by 1
                // until maxDoc is reached. If the document has vector field but no value (== null), the NO_MORE_DOCS
                // is going to be returned by this method.
                while (docId < graphNodeIdToDocMap.getMaxDoc() - 1) {
                    docId++;
                    if (liveNodes.get(docId) && index() != NO_VECTOR) {
                        return docId;
                    }
                }
                docId = NO_MORE_DOCS;

                return docId;
            }

            @Override
            public int advance(int target) throws IOException {
                return slowAdvance(target);
            }
        };
    }

    @Override
    public float[] vectorValue(int i) throws IOException {
        try {
            final VectorFloat<?> vector = vectorFloatValue(i);
            Object backing = vector.get();

            if (backing instanceof float[] array) {
                return array;
            }

            System.out.println("Backing: " + backing.getClass().getName());

            // Tried to flip vectors but once its LE and next time is BE.
//            float[] result = new float[vector.length()];
//            System.out.println("Swap vector: ");
//            for (int j = 0; j < vector.length(); j++) {
//                float value = vector.get(j);
//                int bits = Float.floatToRawIntBits(value);
//                int swappedBits = Integer.reverseBytes(bits);
//                result[j] = Float.intBitsToFloat(swappedBits);
//                System.out.println(result[j]);
//            }
//
//            System.out.println("No swap vector: ");
//            float[] result2 = new float[vector.length()];
//            for (int j = 0; j < vector.length(); j++) {
//                result2[j] = vector.get(j);
//                System.out.println(result2[j]);
//            }

            System.out.println("MemorySegment vector: ");
            System.out.println(Arrays.toString(toFloatArray(backing)));

            return toFloatArray(backing);
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }

    public float[] toFloatArray(Object memorySegment) throws Exception {
        Class<?> memorySegmentClass = Class.forName("java.lang.foreign.MemorySegment");
        Class<?> valueLayoutClass = Class.forName("java.lang.foreign.ValueLayout");
        Class<?> ofFloatClass = Class.forName("java.lang.foreign.ValueLayout$OfFloat");

        Field javaFloatField = valueLayoutClass.getField("JAVA_FLOAT");
        Object javaFloatLayout = javaFloatField.get(null);

        Method toArrayMethod = memorySegmentClass.getMethod("toArray", ofFloatClass);
        return (float[]) toArrayMethod.invoke(memorySegment, javaFloatLayout);
    }

    public VectorFloat<?> vectorValueObject(int i) throws IOException {
        return vectorFloatValue(i);
    }

    @Override
    public FloatVectorValues copy() throws IOException {
        return this;
    }

    @Override
    public VectorScorer scorer(float[] query) throws IOException {
        return new JVectorVectorScorer(this, VECTOR_TYPE_SUPPORT.createFloatVector(query), similarityFunction);
    }

}

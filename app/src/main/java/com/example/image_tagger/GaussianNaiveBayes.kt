package com.example.image_tagger

import kotlinx.coroutines.*
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.ln

class GaussianNaiveBayes(private val coroutineScope: CoroutineScope) {

    private fun calculateConditionalProba(
        x: INDArray, classStatistics: Statistics
    ): Float {
        return with(x.sub(classStatistics.mean)) {
            this.mul(this)
                .div(classStatistics.std)
                .sum()
                .add(Transforms.log(classStatistics.std?.mul(2 * PI)).sumNumber())
                .mul(-0.5)
                .sumNumber()
                .toFloat()
        }
    }

    private suspend fun calculateConditionalProbas(x: INDArray, statistics: List<Statistics>): Map<Int, Float> {
        val (classIds, jobs) = statistics.map { classStatistics ->
            Pair(
                classStatistics.cid,
                coroutineScope.async(Dispatchers.Default) {
                    calculateConditionalProba(x, classStatistics)
                }
            )
        }.unzip()

        return classIds
            .zip(jobs.awaitAll())
            .toMap()
    }

    private suspend fun calculateLikelihoods(x: INDArray, statistics: List<Statistics>): Map<Int, Float> {
        val sampleCounts = statistics.map { classStatistics ->
            Pair(classStatistics.cid, classStatistics.count as Int)
        }.toMap()

        val totalCount = sampleCounts
            .map { it.value }
            .sum()

        val priors = sampleCounts
            .map { classCount ->
                val frequency = classCount.value.toFloat().div(totalCount)
                val prior = if (frequency > 0.0) ln(frequency) else -120.0f
                Pair(classCount.key, prior)
            }
            .toMap()

        val conditionals = calculateConditionalProbas(x, statistics)

        return (priors.keys + conditionals.keys)
            .associateWith { classId ->
                sequenceOf(priors[classId], conditionals[classId])
                    .filterNotNull()
                    .reduce { prior, conditional -> prior + conditional }
            }
    }

    suspend fun calculateProbabilities(x: INDArray, statistics: List<Statistics>): Map<Int, Float> {
        val (classIds, likelihoods) = calculateLikelihoods(x, statistics)
            .map { classLikelihoods ->
                Pair(classLikelihoods.key, classLikelihoods.value)
            }.unzip()

        val probabilities = likelihoods
            .map { abs(it) }
            .maxOrNull()
            ?.let { maxAbsLikelihood ->
                likelihoods
                    .map { likelihood -> (likelihood / maxAbsLikelihood).toDouble()}
                    .toDoubleArray()
                    .let { Transforms.softmax(Nd4j.create(it).mul(500)).castTo(DataType.FLOAT) }
                    .toFloatVector()
                    .toList()
            } ?: listOf()

        return classIds
            .zip(probabilities)
            .toMap()
    }

}

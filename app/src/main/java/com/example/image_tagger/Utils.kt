package com.example.image_tagger

import android.graphics.Bitmap
import java.io.ByteArrayOutputStream
import java.security.MessageDigest
import kotlin.math.round

object Utils {

    enum class Action {
        PICK_IMAGE,
        CAPTURE_IMAGE;
        operator fun invoke(): Int = ordinal
    }

    fun hashMd5(bitmap: Bitmap): String {
        return ByteArrayOutputStream().use { baos ->
            MessageDigest
                    .getInstance("MD5")
                    .apply {
                        update(baos.apply {
                            bitmap.compress(Bitmap.CompressFormat.PNG, 100, this)
                        }.toByteArray())
                    }
                    .digest()
                    .map { hexByte -> hexByte.toInt().and(0xFF).toString(16) }
                    .reduce { acc, s -> "${acc}${s}" }
        }
    }

    fun getScaledSize(bitmap: Bitmap): Triple<Int, Int, Int> {
        val shortSide = maxOf(
                bitmap.width, bitmap.height, Constants.INPUT_WIDTH, Constants.INPUT_HEIGHT
        )
        val width = bitmap.width.toFloat()
        val height = bitmap.height.toFloat()

        return Triple(
            round(width * (shortSide / width)).toInt(),
            round(height * (shortSide / height)).toInt(),
            shortSide
        )
    }

    fun getTopPredictions(
            classProbabilities: Map<Int, Float>,
            threshold: Float = Constants.PROBABILITY_THRESHOLD
    ): List<Int> {
        val probabilities = classProbabilities.toMutableMap()
        val predictedClassIds = mutableListOf<Int>()

        var cumulativeProbability = 0.0f
        while (cumulativeProbability <= threshold) {
            val predictedClassId = probabilities
                .maxByOrNull { it.value }
                ?.key

            predictedClassId?.also {
                predictedClassIds.add(it)
                cumulativeProbability += probabilities[it] as Float
                probabilities[it] = 0.0f
            }
        }

        return predictedClassIds
    }

}

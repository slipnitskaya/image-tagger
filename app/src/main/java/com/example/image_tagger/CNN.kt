package com.example.image_tagger

import android.graphics.Bitmap
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils

class CNN(private val pathToModel: String) {

    private val inputBuffer = Tensor.allocateFloatBuffer(
        Constants.NUM_CHANNELS * Constants.INPUT_HEIGHT * Constants.INPUT_WIDTH
    )

    private val inputTensor = Tensor.fromBlob(
        inputBuffer,
        listOf(1, Constants.NUM_CHANNELS, Constants.INPUT_HEIGHT, Constants.INPUT_WIDTH)
            .map { it.toLong() }
            .toLongArray()
    )

    private val cnn: Module by lazy { Module.load(pathToModel) }

    suspend fun forward(input: Bitmap): FloatArray = withContext(Dispatchers.Default) {
        val (scaledWidth, scaledHeight, shortSide) = Utils.getScaledSize(input)
        TensorImageUtils.bitmapToFloatBuffer(
            Bitmap.createScaledBitmap(input, scaledWidth, scaledHeight, true),
            (scaledWidth - shortSide) / 2,
            (scaledHeight - shortSide) / 2,
            Constants.INPUT_WIDTH,
            Constants.INPUT_HEIGHT,
            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
            TensorImageUtils.TORCHVISION_NORM_STD_RGB,
            inputBuffer,
            0
        )

        val output = cnn.forward(IValue.from(inputTensor))
        val outputTensor = output.toTensor()

        outputTensor.dataAsFloatArray
    }

}

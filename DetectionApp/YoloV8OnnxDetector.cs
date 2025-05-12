using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks; // For Parallel.For

namespace DetectionApp
{
    public class YoloV8OnnxDetector : IDisposable
    {
        private InferenceSession _session;
        private string[] _classNames;
        private readonly int _inputWidth = 640;
        private readonly int _inputHeight = 640;
        private readonly float _confidenceThreshold = 0.5f; // Default confidence threshold
        private readonly float _nmsThreshold = 0.45f;      // Default NMS threshold

        public string[] ClassNames => _classNames;
        public int InputWidth => _inputWidth;
        public int InputHeight => _inputHeight;

        public YoloV8OnnxDetector(string modelPath, string[] classNames, float confidenceThreshold = 0.5f, float nmsThreshold = 0.45f)
        {
            if (string.IsNullOrEmpty(modelPath) || !System.IO.File.Exists(modelPath))
            {
                throw new ArgumentException("Invalid model path provided.", nameof(modelPath));
            }

            try
            {
                // Configure session options (e.g., for GPU execution)
                var sessionOptions = new SessionOptions();
                // Uncomment and modify for CUDA execution if needed
                // sessionOptions.AppendExecutionProvider_CUDA();
                // Uncomment and modify for DirectML execution if needed
                // sessionOptions.AppendExecutionProvider_DML();

                _session = new InferenceSession(modelPath, sessionOptions);
                _classNames = classNames ?? Array.Empty<string>(); // Use provided names or empty array
                _confidenceThreshold = confidenceThreshold;
                _nmsThreshold = nmsThreshold;

                // Optional: Validate input/output names/shapes if needed
                // var inputMeta = _session.InputMetadata;
                // var outputMeta = _session.OutputMetadata;
                // Debug.WriteLine($"Model Input: {string.Join(",", inputMeta.Keys)}");
                // Debug.WriteLine($"Model Output: {string.Join(",", outputMeta.Keys)}");
            }
            catch (Exception ex)
            {
                // Clean up session if initialization failed
                _session?.Dispose();
                throw new InvalidOperationException($"Failed to load ONNX model: {ex.Message}", ex);
            }
        }

        public List<DetectionResult> Detect(Mat image)
        {
            if (_session == null)
            {
                throw new InvalidOperationException("Detector is not initialized or has been disposed.");
            }
            if (image.Empty())
            {
                return new List<DetectionResult>();
            }

            // --- Preprocessing ---
            Mat resizedImage = new Mat();
            Cv2.Resize(image, resizedImage, new Size(_inputWidth, _inputHeight));
            Tensor<float> inputTensor = MatToTensor(resizedImage);

            // --- Inference ---
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(_session.InputMetadata.Keys.First(), inputTensor)
            };

            using (var results = _session.Run(inputs))
            {
                // --- Postprocessing ---
                // Assuming the output tensor shape is [batch_size, num_classes + 4, num_predictions]
                // For YOLOv8, output is often [1, 84, 8400] where 84 = 4 (box) + num_classes (e.g., 80)
                // Or it might be transposed: [1, 8400, 84] - check your specific model export
                var outputTensor = results.First().AsTensor<float>();

                // Check if output is transposed [1, 8400, 84] and transpose if necessary
                // This example assumes [1, 84, 8400] format after potential transpose
                var transposedOutput = TransposeOutput(outputTensor); // Implement TransposeOutput if needed

                return ParseOutput(transposedOutput, image.Size());
            }
        }

        // Helper to convert Mat to Tensor (similar to user's code but adapted)
        private Tensor<float> MatToTensor(Mat mat)
        {
            if (mat.Width != _inputWidth || mat.Height != _inputHeight)
            {
                throw new ArgumentException($"Input Mat must be resized to {_inputWidth}x{_inputHeight}.");
            }

            var tensor = new DenseTensor<float>(new[] { 1, 3, _inputHeight, _inputWidth });
            int channels = mat.Channels(); // Should be 3

            // Ensure Mat is continuous for direct memory access if possible
            if (!mat.IsContinuous())
            {
                mat = mat.Clone(); // Clone to make it continuous
            }

            unsafe // Use unsafe context for faster pointer access
            {
                byte* dataPtr = (byte*)mat.DataPointer;
                int rowStride = (int)mat.Step(); // Bytes per row

                Parallel.For(0, _inputHeight, y =>
                {
                    byte* rowPtr = dataPtr + y * rowStride;
                    for (int x = 0; x < _inputWidth; x++)
                    {
                        int pixelIndex = x * channels;
                        // Assuming BGR format from OpenCV
                        tensor[0, 0, y, x] = rowPtr[pixelIndex + 2] / 255f; // R
                        tensor[0, 1, y, x] = rowPtr[pixelIndex + 1] / 255f; // G
                        tensor[0, 2, y, x] = rowPtr[pixelIndex] / 255f;     // B
                    }
                });
            }
            return tensor;
        }

        // Placeholder for transposing output if needed (e.g., from [1, 8400, 84] to [1, 84, 8400])
        private Tensor<float> TransposeOutput(Tensor<float> outputTensor)
        {
            // Example check: If dimensions are [1, 8400, 84]
            if (outputTensor.Dimensions.Length == 3 && outputTensor.Dimensions[1] > outputTensor.Dimensions[2])
            {
                int numPredictions = outputTensor.Dimensions[1]; // 8400
                int numOutputs = outputTensor.Dimensions[2];     // 84 (4 box + num_classes)
                var transposedTensor = new DenseTensor<float>(new[] { 1, numOutputs, numPredictions });

                // Get DenseTensors to access underlying data easily by index
                var sourceDenseTensor = outputTensor.ToDenseTensor();
                var targetDenseTensor = transposedTensor; // Already a DenseTensor

                // Use standard nested for loops instead of Parallel.For with Spans
                for (int outputIndex = 0; outputIndex < numOutputs; outputIndex++) // Iterate through 84 outputs
                {
                    for (int predIndex = 0; predIndex < numPredictions; predIndex++) // Iterate through 8400 predictions
                    {
                        // Access elements directly using tensor indices
                        // Source index: [0, predIndex, outputIndex]
                        // Target index: [0, outputIndex, predIndex]
                        targetDenseTensor[0, outputIndex, predIndex] =
                            sourceDenseTensor[0, predIndex, outputIndex];
                    }
                }
                return transposedTensor;
            }
            // Assume already in correct [1, 84, 8400] format
            return outputTensor;
        }


        // Parse the output tensor and apply NMS
        private List<DetectionResult> ParseOutput(Tensor<float> outputTensor, Size originalImageSize)
        {
            // Assuming outputTensor shape is [1, num_outputs, num_predictions]
            // where num_outputs = 4 (box) + num_classes
            int numPredictions = outputTensor.Dimensions[2]; // e.g., 8400
            int numOutputs = outputTensor.Dimensions[1];     // e.g., 84
            int numClasses = numOutputs - 4;                 // e.g., 80

            var detectedBoxes = new List<Rect>();
            var detectedScores = new List<float>();
            var detectedClassIndices = new List<int>();

            var outputSpan = outputTensor.ToDenseTensor().Buffer.Span;

            // Scale factors
            float scaleX = (float)originalImageSize.Width / _inputWidth;
            float scaleY = (float)originalImageSize.Height / _inputHeight;

            for (int i = 0; i < numPredictions; i++)
            {
                // Get box coordinates (cx, cy, w, h) and confidence
                // Indices depend on the exact output format after transposition
                float cx = outputSpan[0 * numPredictions + i]; // Box center X
                float cy = outputSpan[1 * numPredictions + i]; // Box center Y
                float w = outputSpan[2 * numPredictions + i];  // Box width
                float h = outputSpan[3 * numPredictions + i];  // Box height

                // Find the class with the highest score
                float maxClassScore = 0f;
                int classId = -1;
                for (int c = 0; c < numClasses; c++)
                {
                    // Index for class score: (4 + c) * numPredictions + i
                    float score = outputSpan[(4 + c) * numPredictions + i];
                    if (score > maxClassScore)
                    {
                        maxClassScore = score;
                        classId = c;
                    }
                }

                // Apply confidence threshold (using the class score as confidence for YOLOv8)
                if (maxClassScore >= _confidenceThreshold)
                {
                    // Convert cx, cy, w, h to x1, y1, x2, y2
                    float x1 = (cx - w / 2);
                    float y1 = (cy - h / 2);
                    float x2 = (cx + w / 2);
                    float y2 = (cy + h / 2);

                    // Scale box to original image size
                    int origX1 = Math.Max(0, (int)(x1 * scaleX));
                    int origY1 = Math.Max(0, (int)(y1 * scaleY));
                    int origX2 = Math.Min(originalImageSize.Width - 1, (int)(x2 * scaleX));
                    int origY2 = Math.Min(originalImageSize.Height - 1, (int)(y2 * scaleY));

                    detectedBoxes.Add(new Rect(origX1, origY1, origX2 - origX1, origY2 - origY1));
                    detectedScores.Add(maxClassScore);
                    detectedClassIndices.Add(classId);
                }
            }

            // Apply Non-Maximum Suppression (NMS) using OpenCV
            int[] nmsIndices;
            if (detectedBoxes.Count > 0)
            {
                CvDnn.NMSBoxes(detectedBoxes, detectedScores, _confidenceThreshold, _nmsThreshold, out nmsIndices);
            }
            else
            {
                nmsIndices = Array.Empty<int>();
            }


            var finalResults = new List<DetectionResult>();
            foreach (int index in nmsIndices)
            {
                string className = (index >= 0 && detectedClassIndices[index] < _classNames.Length)
                                   ? _classNames[detectedClassIndices[index]]
                                   : $"Class_{detectedClassIndices[index]}"; // Fallback if index out of bounds

                finalResults.Add(new DetectionResult(
                    detectedClassIndices[index],
                    className,
                    detectedScores[index],
                    detectedBoxes[index]
                ));
            }

            return finalResults;
        }


        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                if (_session != null)
                {
                    _session.Dispose();
                    _session = null;
                }
            }
        }
    }
}

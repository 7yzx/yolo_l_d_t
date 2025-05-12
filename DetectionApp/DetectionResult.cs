using OpenCvSharp; // Assuming OpenCvSharp is used for Rect

namespace DetectionApp
{
    public class DetectionResult
    {
        public int ClassId { get; set; }
        public string ClassName { get; set; }
        public float Confidence { get; set; }
        public Rect Box { get; set; } // Using OpenCvSharp.Rect

        public DetectionResult(int classId, string className, float confidence, Rect box)
        {
            ClassId = classId;
            ClassName = className ?? $"Class_{classId}"; // Fallback name
            Confidence = confidence;
            Box = box;
        }
    }
}

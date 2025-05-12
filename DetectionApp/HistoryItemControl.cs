using System;
using System.Drawing;
using System.IO;
using System.Windows.Forms;

namespace DetectionApp
{
    public partial class HistoryItemControl : UserControl
    {
        private Image _thumbnail = null; // Store the thumbnail image

        public HistoryItemControl()
        {
            InitializeComponent();
        }

        public void SetData(string imagePath, string pipeClass, int positiveCount, bool isPass)
        {
            try
            {
                // Load thumbnail (important for performance)
                _thumbnail = LoadThumbnail(imagePath, picThumbnail.Width, picThumbnail.Height);
                picThumbnail.Image = _thumbnail;

                // Set filename (show only filename, full path in tooltip)
                lblFilename.Text = Path.GetFileName(imagePath);
                lblFilename.ToolTipText = imagePath; // Show full path on hover

                // Set results text
                string passStatus = isPass ? "合格" : "不合格";
                lblResults.Text = $"类别: {pipeClass} | 螺丝: {positiveCount} | {passStatus}";
                lblResults.ForeColor = isPass ? Color.Green : Color.Red;
            }
            catch (Exception ex)
            {
                lblFilename.Text = "加载错误";
                lblResults.Text = ex.Message;
                picThumbnail.Image = null; // Clear image on error
                _thumbnail?.Dispose(); // Dispose if partially loaded
                _thumbnail = null;
            }
        }

        private Image LoadThumbnail(string imagePath, int targetWidth, int targetHeight)
        {
            if (!File.Exists(imagePath))
            {
                throw new FileNotFoundException("Image file not found.", imagePath);
            }

            // Load the original image
            using (Image originalImage = Image.FromFile(imagePath))
            {
                // Calculate aspect ratio
                float ratioX = (float)targetWidth / originalImage.Width;
                float ratioY = (float)targetHeight / originalImage.Height;
                float ratio = Math.Min(ratioX, ratioY); // Use minimum ratio to fit entirely

                int newWidth = (int)(originalImage.Width * ratio);
                int newHeight = (int)(originalImage.Height * ratio);

                // Create a new bitmap for the thumbnail
                Bitmap thumbnail = new Bitmap(newWidth, newHeight);
                using (Graphics graphics = Graphics.FromImage(thumbnail))
                {
                    // Use high-quality settings for resizing
                    graphics.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
                    graphics.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.HighQuality;
                    graphics.PixelOffsetMode = System.Drawing.Drawing2D.PixelOffsetMode.HighQuality;
                    graphics.CompositingQuality = System.Drawing.Drawing2D.CompositingQuality.HighQuality;

                    graphics.DrawImage(originalImage, 0, 0, newWidth, newHeight);
                }
                return thumbnail;
            }
        }

        // Optional: Add a method to explicitly release the image if needed elsewhere
        public void ReleaseImage()
        {
             picThumbnail.Image = null;
             _thumbnail?.Dispose();
             _thumbnail = null;
        }
    }
}

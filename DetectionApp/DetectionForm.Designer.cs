using DetectionApp;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace YOLO_annotation
{
    public partial class Form3 : Form
    {

        private readonly string _baseOutputPath; // e.g., e:\ALL_CODE\xiangmu\YOLO_train_detection_GUI\output
        private string _selectedProjectModelPath = "";
        private string _selectedProjectClassFilePath = "";
        private string _cameraSaveDir = "";
        private string[] _currentClassNames = Array.Empty<string>();
        private List<string> _classLabels;

        private VideoCapture _capture;
        private CancellationTokenSource _cameraTokenSource;
        private Task _cameraProcessingTask;
        private YoloV8OnnxDetector _detector;

        // Stability Tracking
        private List<DetectionResult> _lastFrameResults = new List<DetectionResult>();
        private int _stableFramesCount = 0;
        private readonly int _stabilityThresholdPx = 15; // Max center point movement
        private readonly int _stableFramesRequired = 5;  // Frames needed for stability
        private readonly double _captureCooldownSeconds = 3.0; // Cooldown after auto capture
        private DateTime _lastAutoCaptureTime = DateTime.MinValue;
        private readonly int _minBoxAreaForStability = 1000; // Min area for stability check
        private int _sceneIdCounter = 0; // For naming saved files

        // UI Update Synchronization
        private readonly object _uiLock = new object(); // Lock for accessing shared UI state if needed
        private readonly int _maxHistoryItems = 20; // Limit history items
        private bool _isAutoSaveEnabled = true; // Field to track auto-save state

        public Form3(string basePath)
        {
            InitializeComponent();
            // base path 是 bin？
            _baseOutputPath = Path.Combine(basePath, "output");
            Directory.CreateDirectory(_baseOutputPath); // Ensure output directory exists
            _isAutoSaveEnabled = chkAutoSave.Checked; // Initialize from checkbox state

        }

        private void Form3_Load(object sender, EventArgs e)
        {
            UpdateProjectDropdown();
            UpdateStatus("就绪 | 按 Enter 手动保存");
            StartCamera(); // Optional: Start camera on load
        }

        private void cmbProjects_DropDown(object sender, EventArgs e)
        {
            UpdateProjectDropdown(); // Refresh list when dropdown is opened
        }

        // 选择项目。首先是默认的项目在 output 目录下default
        private void UpdateProjectDropdown()
        {
            string currentSelection = cmbProjects.SelectedItem?.ToString();
            cmbProjects.Items.Clear();
            cmbProjects.Items.Add("使用默认模型"); // Placeholder

            if (!Directory.Exists(_baseOutputPath))
            {
                cmbProjects.Items.Add("错误: output/ 目录未找到");
                cmbProjects.SelectedIndex = 0;
                return;
            }

            try
            {
                var projects = Directory.GetDirectories(_baseOutputPath)
                                        .Select(Path.GetFileName)
                                        .OrderBy(name => name)
                                        .ToArray();

                if (projects.Length > 0)
                {
                    cmbProjects.Items.AddRange(projects);
                }
                else
                {
                    cmbProjects.Items.Add("无项目 (output/ 下无文件夹)");
                }

                // Try to restore selection
                int indexToSelect = 0; // Default to "Select project..."
                if (!string.IsNullOrEmpty(currentSelection) && currentSelection != "选择项目...")
                {
                    int foundIndex = cmbProjects.FindStringExact(currentSelection);
                    if (foundIndex != -1)
                    {
                        indexToSelect = foundIndex;
                    }
                }
                cmbProjects.SelectedIndex = indexToSelect;
            }
            catch (Exception ex)
            {
                cmbProjects.Items.Add($"读取错误: {ex.Message}");
                cmbProjects.SelectedIndex = 0;
                UpdateStatus($"错误: 无法读取项目列表 - {ex.Message}");
            }
        }


        private void cmbProjects_SelectedIndexChanged(object sender, EventArgs e)
        {
            string selectedProject = cmbProjects.SelectedItem?.ToString();
            bool isCameraRunning = _cameraProcessingTask != null && !_cameraProcessingTask.IsCompleted;

            // Reset paths and class names
            _selectedProjectModelPath = "";
            _selectedProjectClassFilePath = "";
            _currentClassNames = Array.Empty<string>();
            DisposeDetector(); // Dispose old detector if any

            if (cmbProjects.SelectedIndex < 0 || string.IsNullOrEmpty(selectedProject) || selectedProject.StartsWith("无项目") || selectedProject.StartsWith("错误"))
            {
                lblSelectedModel.Text = "模型: 未选择";
                lblSelectedModel.ForeColor = Color.Red;
                if (isCameraRunning) StopCamera(); // Stop camera if model becomes invalid
                ResetDetectionCounts();
                return;
            }
            string projectDir = Path.Combine(_baseOutputPath, selectedProject);
            string modelPath = Path.Combine(projectDir, "weights", "best.onnx"); // Assuming ONNX model
            string classFilePath = Path.Combine(projectDir, "classes.txt");
            if (cmbProjects.SelectedIndex == 0)
            {
                projectDir = Path.Combine(_baseOutputPath, "default");
                modelPath = Path.Combine(projectDir, "weights", "best.onnx"); // Assuming ONNX model
                classFilePath = Path.Combine(projectDir, "classes.txt");
            }
            // 读取类

            bool modelExists = File.Exists(modelPath);
            bool classFileExists = File.Exists(classFilePath);

            if (modelExists)
            {
                _selectedProjectModelPath = modelPath;
                lblSelectedModel.Text = $"模型: {selectedProject}/weights/best.onnx";
                lblSelectedModel.ForeColor = Color.Green;
                UpdateStatus($"已选择项目: {selectedProject}");

                // Load class names
                if (classFileExists)
                {
                    try
                    {
                        _currentClassNames = File.ReadAllLines(classFilePath)
                                               .Select(line => line.Trim())
                                               .Where(line => !string.IsNullOrEmpty(line))
                                               .ToArray();
                        if (_currentClassNames.Length == 0)
                        {
                            UpdateStatus($"警告: {selectedProject} 的 class.txt 为空");
                        }
                        else
                        {
                            UpdateStatus($"已加载 {selectedProject} 的 {_currentClassNames.Length} 个类");
                        }
                    }
                    catch (Exception ex)
                    {
                        _currentClassNames = Array.Empty<string>();
                        MessageBox.Show($"无法读取类文件:\n{classFilePath}\n\n错误: {ex.Message}", "类文件错误", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                        UpdateStatus($"错误: 无法读取 {selectedProject} 的 class.txt");
                    }
                }
                else
                {
                    _currentClassNames = Array.Empty<string>(); // Reset if file not found
                    UpdateStatus($"警告: 未找到 {selectedProject} 的 class.txt");
                }

                // Try to initialize the detector immediately
                try
                {
                    _detector = new YoloV8OnnxDetector(_selectedProjectModelPath, _currentClassNames);
                    UpdateStatus($"模型和类已加载: {selectedProject}");
                    if (isCameraRunning)
                    {
                        // No need to stop/start, detector is replaced on the fly if needed
                        UpdateStatus($"检测器已更新为 {selectedProject}。摄像头仍在运行。");
                    }
                }
                catch (Exception ex)
                {
                    DisposeDetector();
                    lblSelectedModel.Text = $"模型: {selectedProject} (加载失败!)";
                    lblSelectedModel.ForeColor = Color.Red;
                    MessageBox.Show($"加载模型失败:\n{modelPath}\n\n错误: {ex.Message}", "模型加载错误", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    UpdateStatus($"错误: 加载模型失败 - {selectedProject}");
                    //if (isCameraRunning) StopCamera(); // Stop camera if new model failed to load
                }
            }
            else
            {
                lblSelectedModel.Text = $"模型: {selectedProject}/weights/best.onnx (未找到!)";
                lblSelectedModel.ForeColor = Color.Red;
                UpdateStatus($"错误: 未找到模型文件 - {selectedProject}");
                if (isCameraRunning) StopCamera(); // Stop camera if model becomes invalid
            }
            ResetDetectionCounts();
        }

        private void DisposeDetector()
        {
            _detector?.Dispose();
            _detector = null;
        }


        private void btnStartStopCamera_Click(object sender, EventArgs e)
        {
            bool isRunning = _cameraProcessingTask != null && !_cameraProcessingTask.IsCompleted;

            if (isRunning)
            {
                StopCamera();
            }
            else
            {
                StartCamera();
            }
        }
        private void StartCamera()
        {
            if (string.IsNullOrEmpty(_selectedProjectModelPath) || _detector == null)
            {
                MessageBox.Show("请先选择一个有效的项目和模型。", "错误", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }

            //string cameraIdStr = txtCameraId.Text.Trim();
            string cameraIdStr = "0"; // Default to 0 for testing
            if (string.IsNullOrEmpty(cameraIdStr))
            {
                MessageBox.Show("请输入摄像头 ID ", "错误", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }

            // 历史图片都保存
            if (string.IsNullOrEmpty(_cameraSaveDir))
            {
                string selectedProject = cmbProjects.SelectedItem?.ToString();
                if (cmbProjects.SelectedIndex > 0 && !string.IsNullOrEmpty(selectedProject))
                {
                    _cameraSaveDir = Path.Combine(_baseOutputPath, selectedProject, "camera_captures");
                }
                else if (cmbProjects.SelectedIndex == 0)
                {
                    _cameraSaveDir = Path.Combine(_baseOutputPath, "default", "camera_captures");
                }
                else
                {
                    _cameraSaveDir = Path.Combine(Environment.CurrentDirectory, "camera_captures"); // Fallback
                }
                UpdateSaveFolderLabel(); // Update label with default path
            }
            Directory.CreateDirectory(_cameraSaveDir); // Ensure directory exists

            try
            {
                // Try parsing as integer first, otherwise use as string (for URLs)
                if (int.TryParse(cameraIdStr, out int camId))
                {
                    _capture = new VideoCapture(camId);
                }
                else
                {
                    _capture = new VideoCapture(cameraIdStr);
                }

                if (!_capture.IsOpened())
                {
                    throw new Exception($"无法打开摄像头: {cameraIdStr}");
                }

                // Start background processing
                _cameraTokenSource = new CancellationTokenSource();
                var token = _cameraTokenSource.Token;
                _cameraProcessingTask = Task.Run(() => ProcessCameraFrames(token), token);

                // Update UI
                btnStartStopCamera.Text = "STOP";
                btnStartStopCamera.BackColor = Color.Red;
                SetCameraControlsEnabled(false); // Disable controls while running
                UpdateStatus("摄像头启动中...");
                picCameraFeed.Image = null; // Clear previous image
                ResetDetectionCounts();
                _stableFramesCount = 0; // Reset stability on start
                _lastFrameResults.Clear();
            }
            catch (Exception ex)
            {
                MessageBox.Show($"启动摄像头失败: {ex.Message}", "错误", MessageBoxButtons.OK, MessageBoxIcon.Error);
                _capture?.Release();
                _capture = null;
                UpdateStatus($"错误: 启动摄像头失败 - {ex.Message}");
            }
        }

        private void StopCamera()
        {
            if (_cameraTokenSource != null)
            {
                UpdateStatus("正在停止摄像头...");
                _cameraTokenSource.Cancel();
                // Wait briefly for the task to acknowledge cancellation
                // Avoid long waits on UI thread if possible
                // Consider Task.WhenAny with a timeout if needed
                // await _cameraProcessingTask; // Potential deadlock if task waits on UI
            }

            // UI updates should happen after task is likely stopped or in OnCameraStopped
            // OnCameraStopped will be called from the finally block of ProcessCameraFrames
        }

        private void SetCameraControlsEnabled(bool enabled)
        {
            // Use Invoke if called from non-UI thread, but typically called from UI thread here
            if (InvokeRequired)
            {
                Invoke(new Action(() => SetCameraControlsEnabled(enabled)));
                return;
            }
            cmbProjects.Enabled = enabled;
            //txtCameraId.Enabled = enabled;
            btnSelectSaveFolder.Enabled = enabled;
        }


        private void OnCameraStopped()
        {
            // Ensure this runs on the UI thread
            if (InvokeRequired)
            {
                Invoke(new Action(OnCameraStopped));
                return;
            }

            _capture?.Release(); // Release capture resource
            _capture = null;
            _cameraProcessingTask = null; // Clear task reference

            btnStartStopCamera.Text = "开始检测";
            btnStartStopCamera.BackColor = Color.Green;
            SetCameraControlsEnabled(true); // Re-enable controls

            // Clear image or show stopped message
            // picCameraFeed.Image = null; // Optional: clear image
            if (picCameraFeed.Image == null) // Only update text if no image is present
            {
                using (var bmp = new Bitmap(picCameraFeed.Width, picCameraFeed.Height))
                using (var g = Graphics.FromImage(bmp))
                {
                    g.Clear(Color.Black);
                    TextRenderer.DrawText(g, "摄像头已停止", Font, picCameraFeed.ClientRectangle, Color.White, TextFormatFlags.HorizontalCenter | TextFormatFlags.VerticalCenter);
                    // Dispose previous image before assigning new one
                    var oldImage = picCameraFeed.Image;
                    picCameraFeed.Image = (Bitmap)bmp.Clone(); // Assign a clone
                    oldImage?.Dispose();
                }
            }


            // Update status only if not already showing a critical error
            string currentStatus = toolStripStatusLabel1.Text;
            if (!currentStatus.Contains("错误") && !currentStatus.Contains("失败"))
            {
                UpdateStatus("摄像头已停止 | 按 Enter 手动保存");
            }

            _cameraTokenSource?.Dispose();
            _cameraTokenSource = null;
        }

        // --- 帧处理以及检测 ---

        private async Task ProcessCameraFrames(CancellationToken token)
        {
            using (Mat frame = new Mat())
            {
                try
                {
                    // 摄像头读取
                    while (!token.IsCancellationRequested)
                    {
                        if (_capture == null || !_capture.IsOpened())
                        {
                            UpdateStatus("错误: 摄像头连接丢失");
                            break; // Exit loop if capture is lost
                        }
                        // 读取一帧
                        bool success = _capture.Read(frame);
                        if (!success || frame.Empty())
                        {
                            await Task.Delay(10, token); // Wait briefly if frame read fails
                            continue;
                        }
                        Console.WriteLine($"Frame Channels: {frame.Channels()}"); // 🟢 debug打印

                        if (frame.Channels() == 1)
                        {
                            Cv2.CvtColor(frame, frame, ColorConversionCodes.GRAY2BGR);
                        }

                        // --- Detection ---
                        List<DetectionResult> currentResults = new List<DetectionResult>();
                        Bitmap displayBitmap = null;
                        Mat frameForDisplay = frame.Clone(); // 用于显示

                        if (_detector != null)
                        {
                            try
                            {
                                currentResults = _detector.Detect(frame); // 开始检测
                                DrawDetections(frameForDisplay, currentResults); // 画检测结果
                            }
                            catch (ObjectDisposedException)
                            {
                                // Detector might have been disposed by project change, exit loop gracefully
                                UpdateStatus("检测器已更改，正在停止...");
                                break;
                            }
                            catch (Exception ex)
                            {
                                Debug.WriteLine($"Detection error: {ex.Message}");
                                UpdateStatus($"错误: 检测失败 - {ex.Message}");
                                // Optionally draw error message on frame
                                Cv2.PutText(frameForDisplay, $"Detection Error: {ex.Message}", new OpenCvSharp.Point(10, 30), HersheyFonts.HersheySimplex, 0.7, Scalar.Red, 2);
                            }
                        }
                        else
                        {
                            Cv2.PutText(frameForDisplay, "Detector not loaded", new OpenCvSharp.Point(10, 30), HersheyFonts.HersheySimplex, 0.7, Scalar.Yellow, 2);
                        }


                        // --- 是否触发自动保存，如果不需要就注释 ---
                        bool isStable = CheckStability(currentResults);
                        if (isStable && (DateTime.Now - _lastAutoCaptureTime).TotalSeconds > _captureCooldownSeconds)
                        {
                            UpdateStatus($"物体稳定 ({_stableFramesCount} 帧), 自动保存...");
                            // Use original frame for saving, results from detection
                            SaveFrameAndDetections(frame, currentResults, true);
                            _lastAutoCaptureTime = DateTime.Now;
                            _stableFramesCount = 0; // Reset after capture
                            _lastFrameResults.Clear(); // Reset history after capture
                        }
                        else if (isStable)
                        {
                            // Optional: Indicate cooldown active
                            // UpdateStatus($"物体稳定, 冷却中 ({ (DateTime.Now - _lastAutoCaptureTime).TotalSeconds:F1}s / {_captureCooldownSeconds}s)");
                        }

                        // Update last results for next frame's stability check
                        _lastFrameResults = currentResults;

                        // --- 是否触发自动保存，如果不需要就注释 ---

                        // --- UI 更新 ---
                        try
                        {
                            // Convert Mat to Bitmap for display
                            displayBitmap = BitmapConverter.ToBitmap(frameForDisplay);
                            UpdateUI(displayBitmap, currentResults);
                        }
                        catch (Exception ex)
                        {
                            Debug.WriteLine($"UI Update/Bitmap Conversion error: {ex.Message}");
                            // Don't stop processing, just log the error
                        }
                        finally
                        {
                            frameForDisplay.Dispose(); // Dispose the copy used for drawing
                            // displayBitmap is disposed in UpdateUI after assignment
                        }

                        await Task.Delay(10, token); // Small delay to prevent excessive CPU usage
                    }
                }
                catch (OperationCanceledException)
                {
                    // Expected when cancellation is requested
                    UpdateStatus("摄像头处理已取消。");
                }
                catch (Exception ex)
                {
                    // Log unexpected errors during processing
                    Debug.WriteLine($"Camera processing loop error: {ex.Message}");
                    UpdateStatus($"错误: 摄像头处理失败 - {ex.Message}");
                }
                finally
                {
                    // Ensure UI is reset correctly when the loop finishes or is cancelled
                    OnCameraStopped();
                }
            } // Dispose frame Mat
        }

        // ---画检测结果---
        private void DrawDetections(Mat image, List<DetectionResult> results)
        {
            if (results == null) return;

            foreach (var result in results)
            {
                // Draw bounding box
                Cv2.Rectangle(image, result.Box, Scalar.LimeGreen, 2);

                // Prepare label text
                int classId = result.ClassId;
                string label = $"{result.ClassName} {result.Confidence:F2}";

                // Get text size to draw background rectangle
                OpenCvSharp.Size labelSize = Cv2.GetTextSize(label, HersheyFonts.HersheySimplex, 0.5, 1, out int baseline);
                OpenCvSharp.Point labelOrigin = new OpenCvSharp.Point(result.Box.X, result.Box.Y - labelSize.Height - baseline);
                if (labelOrigin.Y < 0) // Adjust if label goes off-screen top
                {
                    labelOrigin.Y = result.Box.Y + baseline; // Place below top edge
                }

                // Draw background rectangle for label
                Cv2.Rectangle(image,
                              new Rect(labelOrigin.X, labelOrigin.Y, labelSize.Width, labelSize.Height + baseline),
                              Scalar.Black, // Background color
                              -1); // Filled rectangle

                // Draw label text
                Cv2.PutText(image, label, new OpenCvSharp.Point(labelOrigin.X, labelOrigin.Y + labelSize.Height),
                            HersheyFonts.HersheySimplex, 0.5, Scalar.LimeGreen, 1);
            }
        }

        private void UpdateUI(Bitmap bitmap, List<DetectionResult> results)
        {
            if (InvokeRequired)
            {
                try
                {
                    // Use BeginInvoke for potentially faster UI responsiveness,
                    // but be mindful of potential race conditions if order matters strictly.
                    BeginInvoke(new Action(() => UpdateUI(bitmap, results)));
                }
                catch (ObjectDisposedException) { /* Form closing, ignore */ }
                catch (InvalidOperationException) { /* Control handle not created, ignore */ }
                return;
            }

            // Update PictureBox
            var oldImage = picCameraFeed.Image;
            picCameraFeed.Image = bitmap; // Assign the new bitmap
            oldImage?.Dispose(); // Dispose the old bitmap *after* assigning the new one

            // Update detection counts and pass/fail status
            UpdateDetectionCountsUI(results);
        }

        // 检测结果
        private void UpdateDetectionCountsUI(List<DetectionResult> results)
        {
            // Ensure called on UI thread (already handled by UpdateUI caller)
            if (results == null) return;

            int countA = results.Count(r => r.ClassName.Equals("A", StringComparison.OrdinalIgnoreCase));
            int countB = results.Count(r => r.ClassName.Equals("B", StringComparison.OrdinalIgnoreCase));
            int positiveCount = results.Count(r => r.ClassName.Equals("positive", StringComparison.OrdinalIgnoreCase));

            string pipeClass = "无";
            if (countA > 0 && countB > 0)
            {
                pipeClass = (countA >= countB) ? "A" : "B";
            }
            else if (countA > 0)
            {
                pipeClass = "A";
            }
            else if (countB > 0)
            {
                pipeClass = "B";
            }

            txtPipeClass.Text = pipeClass;
            txtPositiveCount.Text = positiveCount.ToString();

            if (positiveCount == 4) // Assuming 4 is the target count for "合格"
            {
                lblPassFail.Text = "检测合格";
                lblPassFail.ForeColor = Color.Green;
                lblPassFail.Font = new Font(lblPassFail.Font, FontStyle.Bold);
            }
            else
            {
                lblPassFail.Text = "检测不合格";
                lblPassFail.ForeColor = Color.Red;
                lblPassFail.Font = new Font(lblPassFail.Font, FontStyle.Bold);
            }
        }

        private void ResetDetectionCounts()
        {
            if (InvokeRequired)
            {
                Invoke(new Action(ResetDetectionCounts));
                return;
            }
            txtPipeClass.Text = "无";
            txtPositiveCount.Text = "0";
            lblPassFail.Text = "状态: N/A";
            lblPassFail.ForeColor = Color.Gray;
            lblPassFail.Font = new Font(lblPassFail.Font, FontStyle.Regular);
        }


        // --- Stability and Saving ---
        //检查连续帧检测到的最大目标 中心位置移动距离是否小于阈值，从而判断目标“是否稳定”。
        // 连续 _stableFramesRequired 帧都稳定 → 返回 true。
        private bool CheckStability(List<DetectionResult> currentResults)
        {
            if (currentResults == null || !currentResults.Any() || _lastFrameResults == null || !_lastFrameResults.Any())
            {
                _stableFramesCount = 0;
                return false;
            }

            // Find largest box in current and previous frame (simple approach)
            var largestCurrent = currentResults
                .Where(r => r.Box.Width * r.Box.Height >= _minBoxAreaForStability)
                .OrderByDescending(r => r.Box.Width * r.Box.Height)
                .FirstOrDefault();

            var largestLast = _lastFrameResults
                .Where(r => r.Box.Width * r.Box.Height >= _minBoxAreaForStability)
                .OrderByDescending(r => r.Box.Width * r.Box.Height)
                .FirstOrDefault();

            if (largestCurrent == null || largestLast == null)
            {
                _stableFramesCount = 0;
                return false;
            }

            // Calculate center points
            Point2f currentCenter = new Point2f(largestCurrent.Box.X + largestCurrent.Box.Width / 2f,
                                                largestCurrent.Box.Y + largestCurrent.Box.Height / 2f);
            Point2f lastCenter = new Point2f(largestLast.Box.X + largestLast.Box.Width / 2f,
                                             largestLast.Box.Y + largestLast.Box.Height / 2f);

            // Calculate distance
            double distance = currentCenter.DistanceTo(lastCenter);

            if (distance < _stabilityThresholdPx)
            {
                _stableFramesCount++;
            }
            else
            {
                _stableFramesCount = 0; // Reset if moved too much
            }

            return _stableFramesCount >= _stableFramesRequired;
        }

        private void SaveFrameAndDetections(Mat originalFrame, List<DetectionResult> results, bool automatic)
        {
            if (string.IsNullOrEmpty(_cameraSaveDir))
            {
                UpdateStatus("错误: 未设置保存目录");
                return;
            }
            Directory.CreateDirectory(_cameraSaveDir); // Ensure it exists

            try
            {
                _sceneIdCounter++;
                string timestamp = DateTime.Now.ToString("yyyyMMddHHmmss");
                string baseFilename = $"{timestamp}_{_sceneIdCounter:D4}";
                string statusPrefix = automatic ? "自动保存" : "手动保存";
                string detectionImagePath = ""; // Store path for history

                // 1. 需要保存原图吗？如果需要就注释掉
                //string originImagePath = Path.Combine(_cameraSaveDir, $"{baseFilename}_origin.jpg");
                //if (originalFrame.SaveImage(originImagePath))
                //{
                //    UpdateStatus($"{statusPrefix} 原图: {Path.GetFileName(originImagePath)}");
                //}
                //else
                //{
                //    UpdateStatus($"错误: 保存原图失败 - {Path.GetFileName(originImagePath)}");
                //}


                // 2. 保存检测结果图
                using (Mat detectionFrame = originalFrame.Clone())
                {
                    DrawDetections(detectionFrame, results); // Draw boxes on the copy
                    detectionImagePath = Path.Combine(_cameraSaveDir, $"{baseFilename}_detection.jpg");
                    if (detectionFrame.SaveImage(detectionImagePath))
                    {
                        UpdateStatus($"{statusPrefix} 检测图: {Path.GetFileName(detectionImagePath)}");
                    }
                    else
                    {
                        UpdateStatus($"错误: 保存检测图失败 - {Path.GetFileName(detectionImagePath)}");
                        detectionImagePath = ""; // Clear path on failure

                    }
                }

                // 3. Save Labels (Optional)
                // string txtPath = Path.Combine(_cameraSaveDir, $"{baseFilename}_detection.txt");
                // try
                // {
                //     using (StreamWriter writer = new StreamWriter(txtPath))
                //     {
                //         foreach (var res in results)
                //         {
                //             // Format: class_name confidence x1 y1 x2 y2
                //             writer.WriteLine($"{res.ClassName} {res.Confidence:F4} {res.Box.X} {res.Box.Y} {res.Box.Right} {res.Box.Bottom}");
                //         }
                //     }
                //     UpdateStatus($"{statusPrefix} 标签: {Path.GetFileName(txtPath)}");
                // }
                // catch (Exception ex)
                // {
                //     UpdateStatus($"错误: 保存标签失败 - {ex.Message}");
                // }

                // 4. Add to History Panel (if detection image was saved)
                if (!string.IsNullOrEmpty(detectionImagePath))
                {
                    // Extract results needed for history item
                    int countA = results.Count(r => r.ClassName.Equals("A", StringComparison.OrdinalIgnoreCase));
                    int countB = results.Count(r => r.ClassName.Equals("B", StringComparison.OrdinalIgnoreCase));
                    int positiveCount = results.Count(r => r.ClassName.Equals("positive", StringComparison.OrdinalIgnoreCase));
                    string pipeClass = "N/A";
                    if (countA > 0 && countB > 0) pipeClass = (countA >= countB) ? "A" : "B";
                    else if (countA > 0) pipeClass = "A";
                    else if (countB > 0) pipeClass = "B";
                    bool isPass = positiveCount == 4;
                    int counttest_person = results.Count(r => r.ClassName.Equals("person", StringComparison.OrdinalIgnoreCase));
                    string person_class = "NAN";
                    if (counttest_person > 0)  person_class = "person";
                    // Add to history on UI thread
                    //AddHistoryItem(detectionImagePath, pipeClass, positiveCount, isPass);
                    AddHistoryItem(detectionImagePath, person_class, positiveCount, isPass);
                }


                // Reset stability if saved automatically
                if (automatic)
                {
                    _stableFramesCount = 0;
                    _lastFrameResults.Clear();
                    _lastAutoCaptureTime = DateTime.Now; // Ensure cooldown is based on this capture
                }

                // Reset stability if saved automatically
                if (automatic)
                {
                    _stableFramesCount = 0;
                    _lastFrameResults.Clear();
                    _lastAutoCaptureTime = DateTime.Now; // Ensure cooldown is based on this capture
                }
            }
            catch (Exception ex)
            {
                UpdateStatus($"错误: 保存过程中出错 - {ex.Message}");
                Debug.WriteLine($"Save error: {ex}");
            }
        }
        private void AddHistoryItem(string imagePath, string pipeClass, int positiveCount, bool isPass)
        {
            if (flpHistory.InvokeRequired)
            {
                try
                {
                    flpHistory.Invoke(new Action(() => AddHistoryItem(imagePath, pipeClass, positiveCount, isPass)));
                }
                catch (ObjectDisposedException) { /* Form closing */ }
                catch (InvalidOperationException) { /* Control closing */ }
                return;
            }

            try
            {
                var historyItem = new HistoryItemControl();
                historyItem.SetData(imagePath, pipeClass, positiveCount, isPass);

                // Add to the top of the FlowLayoutPanel
                flpHistory.Controls.Add(historyItem);
                flpHistory.Controls.SetChildIndex(historyItem, 0); // Move to top

                // Limit the number of history items
                while (flpHistory.Controls.Count > _maxHistoryItems)
                {
                    var oldestItem = flpHistory.Controls[flpHistory.Controls.Count - 1] as HistoryItemControl;
                    if (oldestItem != null)
                    {
                        flpHistory.Controls.Remove(oldestItem);
                        oldestItem.ReleaseImage(); // Explicitly release image resource
                        oldestItem.Dispose();      // Dispose the control
                    }
                    else
                    {
                        // Should not happen, but remove the last control anyway
                        flpHistory.Controls.RemoveAt(flpHistory.Controls.Count - 1);
                    }
                }
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Error adding history item: {ex.Message}");
                UpdateStatus($"错误: 添加历史记录失败 - {ex.Message}");
            }
        }
        private void TriggerManualCapture()
        {
            bool isRunning = _cameraProcessingTask != null && !_cameraProcessingTask.IsCompleted;
            if (!isRunning || _capture == null || !_capture.IsOpened() || _detector == null)
            {
                UpdateStatus("摄像头未运行或检测器未加载，无法手动保存。");
                return;
            }

            UpdateStatus("手动保存帧...");
            try
            {
                using (Mat currentFrame = new Mat())
                {
                    // Read a fresh frame directly from the capture device
                    if (_capture.Read(currentFrame) && !currentFrame.Empty())
                    {
                        // Run detection on the captured frame
                        var results = _detector.Detect(currentFrame);
                        // Save it, passing false for automatic
                        SaveFrameAndDetections(currentFrame, results, false);
                    }
                    else
                    {
                        UpdateStatus("手动保存失败: 无法读取当前帧。");
                    }
                }
            }
            catch (Exception ex)
            {
                UpdateStatus($"手动保存失败: {ex.Message}");
                Debug.WriteLine($"Manual capture error: {ex}");
            }
        }

        private void chkAutoSave_CheckedChanged(object sender, EventArgs e)
        {
            _isAutoSaveEnabled = chkAutoSave.Checked;
            UpdateStatus($"自动保存已 {(_isAutoSaveEnabled ? "启用" : "禁用")}");
        }
        // --- UI Event Handlers ---
        // 选择保存文件夹
        private void btnSelectSaveFolder_Click(object sender, EventArgs e)
        {
            // Suggest a path based on selected project if possible
            string selectedProject = cmbProjects.SelectedItem?.ToString();
            if (cmbProjects.SelectedIndex > 0 && !string.IsNullOrEmpty(selectedProject))
            {
                folderBrowserDialog.SelectedPath = Path.Combine(_baseOutputPath, selectedProject, "camera_captures");
            }

            if (folderBrowserDialog.ShowDialog() == DialogResult.OK)
            {
                _cameraSaveDir = folderBrowserDialog.SelectedPath;
                UpdateSaveFolderLabel();
                UpdateStatus($"保存文件夹已更新: {Path.GetFileName(_cameraSaveDir)}");
            }
        }

        private void UpdateStatus(string message)
        {
            // Ensure updates happen on the UI thread
            if (statusStrip1.InvokeRequired)
            {
                try
                {
                    statusStrip1.Invoke(new Action(() => UpdateStatus(message)));
                }
                catch (ObjectDisposedException) { /* Form closing, ignore */ }
                catch (InvalidOperationException) { /* Control handle not created, ignore */ }
                return;
            }
            string timestamp = DateTime.Now.ToString("HH:mm:ss");
            toolStripStatusLabel1.Text = $"状态 ({timestamp}): {message}";
            Debug.WriteLine($"Status ({timestamp}): {message}"); // Also log to debug output
        }


        private void DetectionForm_KeyDown(object sender, KeyEventArgs e)
        {
            // Check if Enter key was pressed
            if (e.KeyCode == Keys.Enter)
            {
                // Check if the focus is not on a button (like Start/Stop) to avoid double actions
                if (!(this.ActiveControl is Button))
                {
                    TriggerManualCapture();
                    e.Handled = true; // Prevent further processing of the Enter key
                    e.SuppressKeyPress = true; // Suppress the 'ding' sound
                }
            }
        }

        private void DetectionForm_FormClosing(object sender, FormClosingEventArgs e)
        {
            StopCamera(); // Request camera stop
            // Give a very short time for the task to potentially cancel, but don't block UI thread for long.
            // Proper async handling would involve awaiting the task completion if possible.
            Task.Delay(100).Wait(); // Small delay, adjust if needed
            CleanupResources(); // Ensure resources are released
        }

        // Centralized resource cleanup
        private void CleanupResources()
        {
            _cameraTokenSource?.Cancel(); // Ensure cancellation is requested
            _capture?.Release();
            _capture = null;
            DisposeDetector(); // Dispose the ONNX session
            _cameraTokenSource?.Dispose();
            _cameraTokenSource = null;
            // Dispose last displayed image if necessary
            // picCameraFeed.Image?.Dispose(); // Be careful if bitmap is shared

            // Dispose history items
            if (flpHistory != null) // Check if panel exists
            {
                // Iterate safely over a copy of the controls collection if modifying it
                var historyControls = flpHistory.Controls.OfType<HistoryItemControl>().ToList();
                foreach (var item in historyControls)
                {
                    flpHistory.Controls.Remove(item); // Remove from panel
                    item.ReleaseImage();
                    item.Dispose();
                }
            }
        }


        private void groupBox1_Enter(object sender, EventArgs e)
        {

        }

        private void statusStrip1_ItemClicked(object sender, ToolStripItemClickedEventArgs e)
        {

        }

        private void txtPipeClass_TextChanged(object sender, EventArgs e)
        {

        }

        private void txtPositiveCount_TextChanged(object sender, EventArgs e)
        {

        }



        private void UpdateSaveFolderLabel()
        {
            if (InvokeRequired)
            {
                Invoke(new Action(UpdateSaveFolderLabel));
                return;
            }

            if (!string.IsNullOrEmpty(_cameraSaveDir))
            {
                // Show only the last part of the path for brevity
                string displayPath = Path.GetFileName(_cameraSaveDir);
                if (string.IsNullOrEmpty(displayPath)) // Handle root directories (e.g., C:\)
                {
                    displayPath = _cameraSaveDir;
                }
                lblSaveFolder.Text = $"保存至: {displayPath}";
                //lblSaveFolder.ToolTipText = _cameraSaveDir; // Show full path on hover
            }
            else
            {
                lblSaveFolder.Text = "保存至: 默认";
                //lblSaveFolder.ToolTipText = "将保存在项目文件夹的 camera_captures 子目录或程序运行目录下";
            }
        }

        private void lblSaveFolder_Click(object sender, EventArgs e)
        {

        }

        private void button1_Click(object sender, EventArgs e)
        {

            if (Directory.Exists(_cameraSaveDir))
            {
                System.Diagnostics.Process.Start("explorer.exe", _cameraSaveDir);
            }
            else
            {
                MessageBox.Show($"文件夹不存在: {_cameraSaveDir}");
            }
        }

    }
}

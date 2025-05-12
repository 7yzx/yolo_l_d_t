using System;
using System.Windows.Forms;

namespace DetectionApp
{
    static class Program
    {
        /// <summary>
        /// The main entry point for the application.
        /// </summary>
        [STAThread]
        static void Main()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            // Determine the base path relative to the executable
            string executablePath = AppDomain.CurrentDomain.BaseDirectory;
            // Navigate up two levels from DetectionApp\bin\Debug (or Release) to YOLO_train_detection_GUI
            string basePath = Path.GetFullPath(Path.Combine(executablePath, @"..\..\"));
            Application.Run(new DetectionForm(basePath));
        }
    }
}

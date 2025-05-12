namespace DetectionApp
{
    partial class HistoryItemControl
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
                // Dispose the thumbnail image explicitly
                _thumbnail?.Dispose();
                _thumbnail = null;
            }
            base.Dispose(disposing);
        }

        #region Component Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.picThumbnail = new System.Windows.Forms.PictureBox();
            this.lblFilename = new System.Windows.Forms.Label();
            this.lblResults = new System.Windows.Forms.Label();
            ((System.ComponentModel.ISupportInitialize)(this.picThumbnail)).BeginInit();
            this.SuspendLayout();
            //
            // picThumbnail
            //
            this.picThumbnail.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.picThumbnail.Location = new System.Drawing.Point(3, 3);
            this.picThumbnail.Name = "picThumbnail";
            this.picThumbnail.Size = new System.Drawing.Size(100, 75); // Adjust size as needed
            this.picThumbnail.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.picThumbnail.TabIndex = 0;
            this.picThumbnail.TabStop = false;
            //
            // lblFilename
            //
            this.lblFilename.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left)
            | System.Windows.Forms.AnchorStyles.Right)));
            this.lblFilename.AutoEllipsis = true;
            this.lblFilename.Location = new System.Drawing.Point(109, 3);
            this.lblFilename.Name = "lblFilename";
            this.lblFilename.Size = new System.Drawing.Size(188, 35); // Adjust size
            this.lblFilename.TabIndex = 1;
            this.lblFilename.Text = "Filename.jpg";
            //
            // lblResults
            //
            this.lblResults.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom)
            | System.Windows.Forms.AnchorStyles.Left)
            | System.Windows.Forms.AnchorStyles.Right)));
            this.lblResults.Location = new System.Drawing.Point(109, 41); // Position below filename
            this.lblResults.Name = "lblResults";
            this.lblResults.Size = new System.Drawing.Size(188, 37); // Adjust size
            this.lblResults.TabIndex = 2;
            this.lblResults.Text = "类别: A | 螺丝: 4 | 合格";
            //
            // HistoryItemControl
            //
            this.AutoScaleDimensions = new System.Drawing.SizeF(7F, 15F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.BackColor = System.Drawing.SystemColors.ControlLight;
            this.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.Controls.Add(this.lblResults);
            this.Controls.Add(this.lblFilename);
            this.Controls.Add(this.picThumbnail);
            this.Margin = new System.Windows.Forms.Padding(3, 3, 3, 0); // Add margin for spacing in FlowLayoutPanel
            this.Name = "HistoryItemControl";
            this.Size = new System.Drawing.Size(300, 81); // Adjust overall size
            ((System.ComponentModel.ISupportInitialize)(this.picThumbnail)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.PictureBox picThumbnail;
        private System.Windows.Forms.Label lblFilename;
        private System.Windows.Forms.Label lblResults;
    }
}

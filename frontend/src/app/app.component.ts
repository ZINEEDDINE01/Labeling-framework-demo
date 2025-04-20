import { Component } from '@angular/core';
import { HttpClient, HttpEventType } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatIconModule } from '@angular/material/icon';
import { MatButtonModule } from '@angular/material/button';

import { ModelManagementComponent } from './model-management/model-management.component';
import { AnnotationComponent } from './annotation/annotation.component';

interface AnnotImage {
  name: string;
  url: string;
  detections?: any;
  annotatedUrl?: string;
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    MatIconModule,
    MatButtonModule,
    ModelManagementComponent,
    AnnotationComponent
  ],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  // New variables for model selection and confidence threshold
  public selectedModelPath: string = '';  // Set this when the user selects a model
  public confidenceThreshold: number = 0.5; // Default confidence threshold

  isUploaded: boolean = false;
  isUpLoading: boolean = false;
  isLoading: boolean = false;
  isAnnotate: boolean = false;
  
  uploadedFiles: any[] = [];
  annotatedFiles: any[] = [];
  
  zoomValue: number = 100;
  imageWidth: number = 200;
  imageHeight: number = 200;
  
  showManual: boolean = false;
  showAuto: boolean = false;
  
  isFileOver: boolean = false;
  
  annotImages: AnnotImage[] = [];
  uploadProgress: number = 0;

  // For custom YOLO training
  customModelTrained: boolean = false;
  customModelPath: string = '';
  showTrainingForm: boolean = false;
    
  // Training hyperparameters (only epochs and model name now)
  epochs: number = 100;
  modelName: string = '';
    
  // For progress bar
  trainingProgress: number = 0;
  
  // Fully annotated count property (for manual annotations, if needed)
  fullyAnnotated: number = 0;
  
  constructor(private http: HttpClient) {}

  onDragOver(event: any) {
    event.preventDefault();
    this.isFileOver = true;
  }
  
  onDragLeave(event: any) {
    event.preventDefault();
    this.isFileOver = false;
  }
  
  onDrop(event: any) {
    event.preventDefault();
    this.isFileOver = false;
    this.onFilesSelected(event);
  }
  
  onFilesSelected(event: any) {
    const files = event.target.files;
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      file.preview = URL.createObjectURL(file);
      this.uploadedFiles.push(file);
      // Save file name and URL for AnnotationComponent
      this.annotImages.push({ name: file.name, url: file.preview });
    }
  }

  uploadImages() {
    if (this.uploadedFiles.length === 0) return;
    
    this.isUpLoading = true;
    const formData = new FormData();
    for (const file of this.uploadedFiles) {
      formData.append('files', file);
    }
    const uploadUrl = 'http://localhost:8000/upload/';
    this.http.post(uploadUrl, formData, {
      reportProgress: true,
      observe: 'events'
    }).subscribe({
      next: (event) => {
        if (event.type === HttpEventType.UploadProgress && event.total) {
          this.uploadProgress = Math.round((100 * event.loaded) / event.total);
        } else if (event.type === HttpEventType.Response) {
          console.log('Upload successful:', event.body);
          this.isUpLoading = false;
          this.isUploaded = true;
        }
      },
      error: (err) => {
        console.error('Upload failed:', err);
        this.isUpLoading = false;
      }
    });
  }

  annotate() {
    this.isLoading = true;
    setTimeout(() => {
      this.annotatedFiles = [...this.uploadedFiles];
      this.isLoading = false;
      this.isAnnotate = true;
    }, 2000);
  }
  
  zoomIn() {
    this.zoomValue += 10;
  }
  
  zoomOut() {
    this.zoomValue -= 10;
  }
  
  isImageRemoved(file: any): boolean {
    return false;
  }
  
  removeFile(file: any) {
    const index = this.uploadedFiles.indexOf(file);
    if (index !== -1) {
      this.uploadedFiles.splice(index, 1);
      this.annotImages.splice(index, 1);
    }
  }
  
  // Run auto annotation by sending the selected model path and confidence threshold to the backend
  runAutoAnnotate() {
    if (!this.customModelTrained || !this.customModelPath) {
      alert("Please train a custom YOLO model first.");
      return;
    }
    console.log("Running auto annotation on dataset using model:", this.customModelPath);
    this.isLoading = true;
    const payload = {
      model_path: this.customModelPath,
      confidence_threshold: this.confidenceThreshold
    };
    this.http.post('http://localhost:8000/api/ai/annotate/dataset', payload)
      .subscribe({
        next: (res: any) => {
          console.log("Auto annotation complete:", res);
          // For each returned annotated image, update the corresponding image in annotImages.
          for (let annotated of res.annotated_images) {
            let img = this.annotImages.find(i => i.name === annotated.filename);
            if (img) {
              img.detections = annotated.detections;
              img.annotatedUrl = annotated.annotated_image; // URL to the annotated image on the backend.
            }
          }
          this.isLoading = false;
          this.isAnnotate = true;  // Show AnnotationComponent with updated detections.
        },
        error: (err) => {
          console.error("Auto annotation failed:", err);
          this.isLoading = false;
        }
      });
  }
  
  // Method to show training form
  trainCustomModel() {
    this.showTrainingForm = true;
  }
  
  // Submit training parameters and simulate a progress bar.
  submitTraining() {
    const payload = {
      epochs: this.epochs,
      model_name: this.modelName
    };
    console.log("Training started with payload:", payload);
    
    this.trainingProgress = 0;
    const interval = setInterval(() => {
      this.trainingProgress += 10;
      if (this.trainingProgress >= 90) {
        clearInterval(interval);
      }
    }, 500);

    this.http.post('http://localhost:8000/api/models/train/yolo', payload)
      .subscribe({
        next: (res: any) => {
          this.trainingProgress = 100;
          this.customModelPath = res.model_path;
          this.customModelTrained = true;
          this.showTrainingForm = false;
          console.log("Training completed. Model path:", this.customModelPath);
        },
        error: (err) => {
          console.error("Training failed:", err);
          clearInterval(interval);
        }
      });
  }
  
  autoAnnotateDataset() {
    this.runAutoAnnotate();
  }
  
  onFullyAnnotatedCount(count: number) {
    console.log("Fully annotated count:", count);
    this.fullyAnnotated = count;
  }
  
  title = 'deepl-label';
}

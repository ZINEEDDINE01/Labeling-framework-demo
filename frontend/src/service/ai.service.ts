import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class AiService {
  // Make sure this URL matches your backend's address and port
  private baseUrl = 'http://localhost:8000';

  constructor(private http: HttpClient) {}


    // Upload images (if separate from file upload component)
    uploadImages(formData: FormData): Observable<any> {
      // Calls the FastAPI /upload/ endpoint
      return this.http.post(`${this.baseUrl}/upload/`, formData);
    }


   // YOLO detection on a single image
  detectYOLO(file: File): Observable<any> {
    const formData = new FormData();
    formData.append('file', file);
    return this.http.post(`${this.baseUrl}/api/ai/detect/yolo`, formData);
  }
 // SAM segmentation on a single image
 segmentSAM(file: File, promptType: string, prompt: number[]): Observable<any> {
  const formData = new FormData();
  formData.append('file', file);
  const samRequest = { prompt_type: promptType, prompt: prompt };
  formData.append('request', new Blob([JSON.stringify(samRequest)], { type: 'application/json' }));
  return this.http.post(`${this.baseUrl}/api/ai/segment/sam`, formData);
}

  // Train a custom YOLO model (with classes)
  trainYOLO(
    trainDataset: string,
    valDataset: string,
    epochs: number,
    modelName: string,
    classes: string[]
  ): Observable<any> {
    return this.http.post(`${this.baseUrl}/api/models/train/yolo`, {
      train_dataset: trainDataset,
      val_dataset: valDataset,
      epochs: epochs,
      model_name: modelName,
      classes: classes
    });
  }

 // Deploy a custom model
  deployYOLO(modelPath: string): Observable<any> {
    return this.http.post(`${this.baseUrl}/api/models/deploy/yolo`, { model_path: modelPath });
  }


  // Example for getting available models (if you implement the endpoint)
  getAvailableModels(): Observable<any> {
    return this.http.get(`${this.baseUrl}/api/models/available`);
  }

  // Auto-annotate the entire dataset using a deployed model
  annotateDataset(modelPath: string): Observable<any> {
    return this.http.post(`${this.baseUrl}/api/ai/annotate/dataset`, { model_path: modelPath });
  }


}

import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class AnnotateService {

  private apiUrl = 'http://127.0.0.1:8000';  // Change to your FastAPI backend URL

  constructor(private http: HttpClient) {}

  // Upload files
  uploadFiles(files: File[]): Observable<any> {
    const formData = new FormData();
    files.forEach(file => {
      formData.append('files', file, file.name);
    });
    console.log("Service called...");
    

    return this.http.post(`${this.apiUrl}/upload/`, formData);
  }

  // Download all files
  downloadAllFiles(): Observable<any> {
    return this.http.get(`${this.apiUrl}/annotate/`);
  }

  // Download a specific file by filename
  downloadFile(filename: string): Observable<Blob> {
    return this.http.get(`${this.apiUrl}/download/${filename}`, {
      responseType: 'blob',
    });
  }
}

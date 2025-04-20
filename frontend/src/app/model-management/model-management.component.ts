import { Component, OnInit } from '@angular/core';
import { AiService } from 'src/service/ai.service';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-model-management',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './model-management.component.html',
  styleUrls: ['./model-management.component.scss']
})
export class ModelManagementComponent implements OnInit {
  availableModels: any[] = [];
  trainingStatus: string = '';
  deployStatus: string = '';

  // For training form
  trainDataset: string = '';
  valDataset: string = '';
  epochs: number = 50;
  modelName: string = '';
  classesInput: string = ''; // Comma-separated string

  // For deploy form
  modelPath: string = '';

  constructor(private aiService: AiService) {}

  ngOnInit(): void {
    this.fetchAvailableModels();
  }

  fetchAvailableModels() {
    this.aiService.getAvailableModels().subscribe(
      (res: any) => {
        this.availableModels = res.models;
      },
      err => {
        console.error('Error fetching models', err);
      }
    );
  }


}

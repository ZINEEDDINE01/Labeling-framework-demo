import { Component, ElementRef, ViewChild, AfterViewInit, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HttpClient, HttpClientModule } from '@angular/common/http';

interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
  label: string;
}

interface ImageInfo {
  name: string;
  url: string;
}

@Component({
  selector: 'app-annotation',
  standalone: true,
  imports: [CommonModule, HttpClientModule],
  templateUrl: './annotation.component.html',
  styleUrls: ['./annotation.component.scss']
})
export class AnnotationComponent implements AfterViewInit {
  @ViewChild('canvas') canvasRef!: ElementRef<HTMLCanvasElement>;

  // Input array of images (each with a name and URL)
  @Input() images: ImageInfo[] = [];
  // Emit the count of images with at least one bounding box
  @Output() fullyAnnotatedCount = new EventEmitter<number>();

  currentIndex: number = 0;

  // Store bounding boxes for each image, keyed by image name
  boxesByImage: { [key: string]: BoundingBox[] } = {};

  // Store dimensions for each image
  imageDimensions: { [key: string]: { width: number; height: number } } = {};

  private ctx!: CanvasRenderingContext2D;
  private image: HTMLImageElement = new Image();

  // State for drawing a new box
  isDrawing: boolean = false;
  startX: number = 0;
  startY: number = 0;
  currentX: number = 0;
  currentY: number = 0;

  // State for moving an existing box
  moving: boolean = false;
  selectedBoxIndex: number | null = null;
  dragOffsetX: number = 0;
  dragOffsetY: number = 0;

  constructor(private http: HttpClient) {}

  ngAfterViewInit(): void {
    if (this.images.length > 0) {
      this.loadImage(this.currentImage.url);
    }
  }

  // Getter for current image info
  get currentImage(): ImageInfo {
    return this.images[this.currentIndex];
  }

  // Get bounding boxes for the current image
  get currentBoxes(): BoundingBox[] {
    const name = this.currentImage.name;
    if (!this.boxesByImage[name]) {
      this.boxesByImage[name] = [];
    }
    return this.boxesByImage[name];
  }

  loadImage(url: string) {
    const canvas = this.canvasRef.nativeElement;
    this.ctx = canvas.getContext('2d') as CanvasRenderingContext2D;
    this.image = new Image();
    this.image.src = url;
    this.image.onload = () => {
      canvas.width = this.image.width;
      canvas.height = this.image.height;
      // Save dimensions for current image
      this.imageDimensions[this.currentImage.name] = {
        width: canvas.width,
        height: canvas.height
      };
      this.redraw();
      // Update annotation count when image loads
      this.updateFullyAnnotatedCount();
    };
  }

  redraw() {
    const canvas = this.canvasRef.nativeElement;
    this.ctx.clearRect(0, 0, canvas.width, canvas.height);
    this.ctx.drawImage(this.image, 0, 0);
    for (let i = 0; i < this.currentBoxes.length; i++) {
      const box = this.currentBoxes[i];
      this.drawBox(box, i === this.selectedBoxIndex);
    }
  }

  drawBox(box: BoundingBox, isSelected: boolean) {
    this.ctx.lineWidth = 2;
    this.ctx.strokeStyle = isSelected ? 'yellow' : 'red';
    this.ctx.strokeRect(box.x, box.y, box.width, box.height);
    this.ctx.font = '16px Arial';
    this.ctx.fillStyle = isSelected ? 'yellow' : 'red';
    const labelY = box.y - 5 < 0 ? box.y + 15 : box.y - 5;
    this.ctx.fillText(box.label, box.x, labelY);
  }

  onMouseDown(event: MouseEvent) {
    const rect = this.canvasRef.nativeElement.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const idx = this.findBoxAtCoords(x, y);
    if (idx !== null) {
      // Start moving the box
      this.selectedBoxIndex = idx;
      this.moving = true;
      const box = this.currentBoxes[idx];
      this.dragOffsetX = x - box.x;
      this.dragOffsetY = y - box.y;
    } else {
      // Start drawing a new box
      this.isDrawing = true;
      this.startX = x;
      this.startY = y;
      this.selectedBoxIndex = null;
    }
  }

  onMouseMove(event: MouseEvent) {
    const rect = this.canvasRef.nativeElement.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    if (this.moving && this.selectedBoxIndex !== null) {
      const box = this.currentBoxes[this.selectedBoxIndex];
      box.x = x - this.dragOffsetX;
      box.y = y - this.dragOffsetY;
      this.redraw();
    } else if (this.isDrawing) {
      this.currentX = x;
      this.currentY = y;
      this.redraw();
      this.ctx.strokeStyle = 'blue';
      this.ctx.lineWidth = 2;
      const w = this.currentX - this.startX;
      const h = this.currentY - this.startY;
      this.ctx.strokeRect(this.startX, this.startY, w, h);
    }
  }

  onMouseUp(event: MouseEvent) {
    if (this.moving) {
      this.moving = false;
    } else if (this.isDrawing) {
      this.isDrawing = false;
      const rect = this.canvasRef.nativeElement.getBoundingClientRect();
      const endX = event.clientX - rect.left;
      const endY = event.clientY - rect.top;
      const w = endX - this.startX;
      const h = endY - this.startY;
      if (Math.abs(w) < 5 || Math.abs(h) < 5) {
        this.redraw();
        return;
      }
      const label = window.prompt('Enter class label for this bounding box:');
      if (label) {
        this.currentBoxes.push({ x: this.startX, y: this.startY, width: w, height: h, label });
      }
    }
    this.redraw();
    this.updateFullyAnnotatedCount();
  }

  findBoxAtCoords(x: number, y: number): number | null {
    for (let i = this.currentBoxes.length - 1; i >= 0; i--) {
      const box = this.currentBoxes[i];
      if (x >= box.x && x <= box.x + box.width && y >= box.y && y <= box.y + box.height) {
        return i;
      }
    }
    return null;
  }

  deleteSelectedBox() {
    if (this.selectedBoxIndex !== null) {
      this.currentBoxes.splice(this.selectedBoxIndex, 1);
      this.selectedBoxIndex = null;
      this.redraw();
      this.updateFullyAnnotatedCount();
    } else {
      window.alert("No bounding box is selected.");
    }
  }

  // New: Save current image annotations to the backend
  saveCurrentAnnotations() {
    if (this.currentBoxes.length === 0) return; // if no annotations, nothing to save
    const payload = {
      filename: this.currentImage.name,
      annotations: this.currentBoxes
    };
    this.http.post('http://localhost:8000/annotations/save', payload)
      .subscribe({
        next: (res) => {
          console.log("Annotations saved for image", this.currentImage.name, res);
          this.updateFullyAnnotatedCount();
        },
        error: (err) => {
          console.error("Error saving annotations for", this.currentImage.name, err);
        }
      });
  }

  // Navigation: Next/Prev image. Save current annotations before switching.
  nextImage() {
    if (this.currentIndex < this.images.length - 1) {
      this.saveCurrentAnnotations();
      this.currentIndex++;
      this.loadImage(this.currentImage.url);
    }
  }

  prevImage() {
    if (this.currentIndex > 0) {
      this.saveCurrentAnnotations();
      this.currentIndex--;
      this.loadImage(this.currentImage.url);
    }
  }

  // Emit the count of images with at least one bounding box (or saved annotation)
  updateFullyAnnotatedCount() {
    let count = 0;
    for (const img of this.images) {
      const name = img.name;
      if (this.boxesByImage[name] && this.boxesByImage[name].length > 0) {
        count++;
      }
    }
    this.fullyAnnotatedCount.emit(count);
  }



  // Export annotations in COCO JSON format
  exportCocoAnnotations() {
    let imageId = 1;
    let annotationId = 1;
    const images: any[] = [];
    const annotations: any[] = [];
    const categoryMap: { [key: string]: number } = {};
    let nextCategoryId = 1;

    for (const img of this.images) {
      const dims = this.imageDimensions[img.name] || { width: 0, height: 0 };
      images.push({
        id: imageId,
        file_name: img.name,
        width: dims.width,
        height: dims.height
      });
      const boxes = this.boxesByImage[img.name] || [];
      for (const box of boxes) {
        if (!(box.label in categoryMap)) {
          categoryMap[box.label] = nextCategoryId++;
        }
        const catId = categoryMap[box.label];
        annotations.push({
          id: annotationId++,
          image_id: imageId,
          category_id: catId,
          bbox: [box.x, box.y, box.width, box.height],
          area: Math.abs(box.width * box.height),
          segmentation: [[box.x, box.y, box.x + box.width, box.y, box.x + box.width, box.y + box.height, box.x, box.y + box.height]],
          iscrowd: 0
        });
      }
      imageId++;
    }
    const categories = Object.keys(categoryMap).map(label => ({
      id: categoryMap[label],
      name: label,
      supercategory: ''
    }));
    const cocoJson = {
      info: {
        description: "Manual Annotations",
        version: "1.0",
        year: new Date().getFullYear(),
        contributor: "",
        date_created: new Date().toISOString()
      },
      licenses: [],
      images: images,
      annotations: annotations,
      categories: categories
    };
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(cocoJson, null, 2));
    const downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", "annotations_coco.json");
    document.body.appendChild(downloadAnchorNode);
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
  }

  // Export function (alias)
  exportAnnotations() {
    this.exportCocoAnnotations();
  }
}

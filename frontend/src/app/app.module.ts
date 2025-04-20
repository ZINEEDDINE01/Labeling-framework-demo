import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { AppComponent } from './app.component';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { MatSidenavModule } from '@angular/material/sidenav';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatBadgeModule } from '@angular/material/badge';
import { RouterModule } from '@angular/router';
import { AnnotateService } from 'src/service/annotate.service';
import { HttpClientModule } from '@angular/common/http';
import { ModelManagementComponent } from './model-management/model-management.component';
import { AnnotationComponent } from './annotation/annotation.component';
@NgModule({
  declarations: [
  
  ],
  imports: [
    BrowserModule,
    ReactiveFormsModule,
    BrowserAnimationsModule,
    MatSidenavModule,
    MatButtonModule,
    MatIconModule,
    MatBadgeModule,
    FormsModule,
    RouterModule, 
    HttpClientModule,
     // Import your standalone components here:
     AppComponent,
     ModelManagementComponent,
     AnnotationComponent
  ],
  providers: [AnnotateService],
  bootstrap: [AppComponent],
})
export class AppModule {}


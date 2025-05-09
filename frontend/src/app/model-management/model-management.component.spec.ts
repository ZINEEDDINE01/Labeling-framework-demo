import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ModelManagementComponent } from './model-management.component';

describe('ModelManagementComponent', () => {
  let component: ModelManagementComponent;
  let fixture: ComponentFixture<ModelManagementComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ModelManagementComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ModelManagementComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});

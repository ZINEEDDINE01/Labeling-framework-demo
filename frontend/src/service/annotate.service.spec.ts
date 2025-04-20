import { TestBed } from '@angular/core/testing';

import { AnnotateService } from './annotate.service';

describe('AnnotateService', () => {
  let service: AnnotateService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(AnnotateService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});

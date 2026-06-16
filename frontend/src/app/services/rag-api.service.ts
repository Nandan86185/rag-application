import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface ChatResponse {
  answer: string;
  sources_used: number;
}

export interface AgentStep {
  tool: string;
  input: string;
}

export interface AgentResponse {
  answer: string;
  steps: AgentStep[];
  sources_used: number;
}

export interface UploadResponse {
  message: string;
  chunks_added: number;
}

@Injectable({
  providedIn: 'root'
})
export class RagApiService {
  private apiUrl = 'http://127.0.0.1:8000';

  constructor(private http: HttpClient) { }

  uploadDocument(file: File): Observable<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    return this.http.post<UploadResponse>(`${this.apiUrl}/upload`, formData);
  }

  chatWithDocument(query: string): Observable<ChatResponse> {
    return this.http.post<ChatResponse>(`${this.apiUrl}/chat`, { query });
  }

  agentChat(query: string): Observable<AgentResponse> {
    return this.http.post<AgentResponse>(`${this.apiUrl}/agent/chat`, { query });
  }
}

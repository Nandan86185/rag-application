import { Component, ElementRef, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { RagApiService, UploadResponse, AgentResponse, AgentStep } from './services/rag-api.service';

interface Message {
  role: 'user' | 'assistant';
  text: string;
  sources?: number;
  steps?: AgentStep[];
  mode?: 'agent';
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss'
})
export class AppComponent {
  title = 'OmniMind';

  // Upload state
  selectedFile: File | null = null;
  isUploading = false;
  uploadMessage = '';
  documentReady = false;  // used for UI only (upload section collapse)

  // Chat state
  messages: Message[] = [
    { role: 'assistant', text: 'Hello! I can answer questions from an uploaded document, search the web, or do both. How can I help you today?' }
  ];
  currentQuery = '';
  isTyping = false;

  @ViewChild('chatContainer') private chatContainer!: ElementRef;

  constructor(private ragApi: RagApiService) { }

  onFileSelected(event: any) {
    const file: File = event.target.files[0];
    if (file) {
      this.selectedFile = file;
    }
  }

  uploadDocument() {
    if (!this.selectedFile) return;

    this.isUploading = true;
    this.uploadMessage = 'Analyzing and extracting knowledge...';

    this.ragApi.uploadDocument(this.selectedFile).subscribe({
      next: (res: UploadResponse) => {
        this.isUploading = false;
        this.documentReady = true;
        this.uploadMessage = `Success! Abstracted ${res.chunks_added} chunks of knowledge.`;
        this.messages.push({ role: 'assistant', text: `I've successfully read "${this.selectedFile!.name}". What would you like to know?` });
        this.scrollToBottom();
      },
      error: (err: any) => {
        this.isUploading = false;
        this.uploadMessage = 'Upload failed. Please check backend logs.';
        console.error(err);
      }
    });
  }

  sendMessage() {
    if (!this.currentQuery.trim()) return;

    const userText = this.currentQuery;
    this.messages.push({ role: 'user', text: userText });
    this.currentQuery = '';
    this.scrollToBottom();

    this.isTyping = true;
    this.scrollToBottom();

    this.ragApi.agentChat(userText).subscribe({
      next: (res: AgentResponse) => {
        this.isTyping = false;
        this.messages.push({
          role: 'assistant',
          text: res.answer,
          steps: res.steps,
          sources: res.sources_used,
          mode: 'agent'
        });
        this.scrollToBottom();
      },
      error: (err: any) => {
        this.isTyping = false;
        this.messages.push({ role: 'assistant', text: '⚠️ Agent error. Please check the backend logs.' });
        this.scrollToBottom();
      }
    });
  }

  scrollToBottom() {
    setTimeout(() => {
      if (this.chatContainer) {
        this.chatContainer.nativeElement.scrollTop = this.chatContainer.nativeElement.scrollHeight;
      }
    }, 100);
  }
}

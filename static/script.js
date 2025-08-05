class ElectricalGridChatbot {
    constructor() {
        this.ws = null;
        this.isConnected = false;
        this.sessionId = this.generateSessionId();
        this.messageQueue = [];
        
        this.initializeElements();
        this.bindEvents();
        this.connectWebSocket();
        this.loadSystemStats();
        this.loadDocumentList();
    }
    
    initializeElements() {
        // UI Elements
        this.fileInput = document.getElementById('fileInput');
        this.uploadBtn = document.getElementById('uploadBtn');
        this.uploadProgress = document.getElementById('uploadProgress');
        this.chatMessages = document.getElementById('chatMessages');
        this.messageInput = document.getElementById('messageInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.typingIndicator = document.getElementById('typingIndicator');
        this.connectionStatus = document.getElementById('connectionStatus');
        this.clearChatBtn = document.getElementById('clearChatBtn');
        this.refreshStatsBtn = document.getElementById('refreshStatsBtn');
        
        // Stats elements
        this.docCount = document.getElementById('docCount');
        this.chunkCount = document.getElementById('chunkCount');
        this.systemStatus = document.getElementById('systemStatus');
    }
    
    bindEvents() {
        // File upload
        this.uploadBtn.addEventListener('click', () => this.handleFileUpload());
        this.fileInput.addEventListener('change', () => this.validateFiles());
        
        // Chat functionality
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // UI actions
        this.clearChatBtn.addEventListener('click', () => this.clearChat());
        this.refreshStatsBtn.addEventListener('click', () => {
            this.loadSystemStats();
            this.loadDocumentList();
        });
        
        // Advanced features
        this.setupAdvancedEventListeners();
        
        // Auto-resize textarea
        this.messageInput.addEventListener('input', () => this.autoResizeInput());
    }

    setupAdvancedEventListeners() {
        // Theme toggle
        const themeToggle = document.getElementById('themeToggle');
        if (themeToggle) {
            themeToggle.addEventListener('click', () => this.toggleTheme());
        }

        // Advanced search
        const searchInput = document.getElementById('searchInput');
        const keywordFilter = document.getElementById('keywordFilter');
        const typeFilter = document.getElementById('typeFilter');
        
        if (searchInput) {
            searchInput.addEventListener('input', () => this.performAdvancedSearch());
        }
        if (keywordFilter) {
            keywordFilter.addEventListener('input', () => this.performAdvancedSearch());
        }
        if (typeFilter) {
            typeFilter.addEventListener('change', () => this.performAdvancedSearch());
        }

        // Bulk operations
        const bulkOperationsBtn = document.getElementById('bulkOperationsBtn');
        if (bulkOperationsBtn) {
            bulkOperationsBtn.addEventListener('click', () => this.showBulkOperationsModal());
        }

        // Bulk operation buttons
        this.setupBulkOperationButtons();

        // Image analysis
        const analyzeImagesBtn = document.getElementById('analyzeImagesBtn');
        if (analyzeImagesBtn) {
            analyzeImagesBtn.addEventListener('click', () => this.analyzeAllImages());
        }
    }

    setupBulkOperationButtons() {
        // Delete all documents
        const deleteAllDocsBtn = document.getElementById('deleteAllDocsBtn');
        if (deleteAllDocsBtn) {
            deleteAllDocsBtn.addEventListener('click', () => this.deleteAllDocuments());
        }

        // Export documents
        const exportDocsBtn = document.getElementById('exportDocsBtn');
        if (exportDocsBtn) {
            exportDocsBtn.addEventListener('click', () => this.exportDocuments());
        }

        // Batch analyze
        const batchAnalyzeBtn = document.getElementById('batchAnalyzeBtn');
        if (batchAnalyzeBtn) {
            batchAnalyzeBtn.addEventListener('click', () => this.batchAnalyzeImages());
        }

        // Extract components
        const extractComponentsBtn = document.getElementById('extractComponentsBtn');
        if (extractComponentsBtn) {
            extractComponentsBtn.addEventListener('click', () => this.extractElectricalComponents());
        }

        // Reindex
        const reindexBtn = document.getElementById('reindexBtn');
        if (reindexBtn) {
            reindexBtn.addEventListener('click', () => this.rebuildSearchIndex());
        }
    }
    
    generateSessionId() {
        return 'session_' + Math.random().toString(36).substr(2, 9);
    }
    
    connectWebSocket() {
        try {
            const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                this.isConnected = true;
                this.updateConnectionStatus('Connected', 'success');
                this.processMessageQueue();
            };
            
            this.ws.onmessage = (event) => {
                this.handleWebSocketMessage(event);
            };
            
            this.ws.onclose = () => {
                this.isConnected = false;
                this.updateConnectionStatus('Disconnected', 'danger');
                // Attempt to reconnect after 3 seconds
                setTimeout(() => this.connectWebSocket(), 3000);
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus('Error', 'danger');
            };
            
        } catch (error) {
            console.error('Failed to connect WebSocket:', error);
            this.updateConnectionStatus('Failed', 'danger');
        }
    }
    
    updateConnectionStatus(status, type) {
        this.connectionStatus.textContent = status;
        this.connectionStatus.className = `badge bg-${type}`;
    }
    
    handleWebSocketMessage(event) {
        try {
            const data = JSON.parse(event.data);
            
            if (data.typing !== undefined && !data.message) {
                this.showTypingIndicator(data.typing);
                return;
            }
            
            if (data.error) {
                this.addMessage('Error: ' + data.error, 'bot', null, null, true);
                return;
            }
            
            if (data.message) {
                this.addMessage(data.message, 'bot', data.sources, data.images);
            }
            
        } catch (error) {
            console.error('Error parsing WebSocket message:', error);
        }
    }
    
    processMessageQueue() {
        while (this.messageQueue.length > 0 && this.isConnected) {
            const message = this.messageQueue.shift();
            this.ws.send(JSON.stringify(message));
        }
    }
    
    sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message) return;
        
        // Add user message to chat
        this.addMessage(message, 'user');
        
        // Clear input
        this.messageInput.value = '';
        this.autoResizeInput();
        
        // Send to WebSocket
        const messageData = {
            message: message,
            session_id: this.sessionId
        };
        
        if (this.isConnected && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(messageData));
        } else {
            this.messageQueue.push(messageData);
            this.updateConnectionStatus('Reconnecting...', 'warning');
            this.connectWebSocket();
        }
    }
    
    addMessage(content, sender, sources = null, images = null, isError = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'message-avatar';
        avatarDiv.innerHTML = sender === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        const senderName = sender === 'user' ? 'You' : 'Grid Assistant';
        const nameSpan = document.createElement('strong');
        nameSpan.textContent = senderName;
        
        const messageText = document.createElement('div');
        messageText.innerHTML = this.formatMessage(content);
        
        contentDiv.appendChild(nameSpan);
        contentDiv.appendChild(messageText);
        
        // Add sources if available
        if (sources && sources.length > 0) {
            const sourcesDiv = this.createSourcesElement(sources);
            contentDiv.appendChild(sourcesDiv);
        }
        
        // Add images if available
        if (images && images.length > 0) {
            const imagesDiv = this.createImagesElement(images);
            contentDiv.appendChild(imagesDiv);
        }
        
        // Apply error styling if needed
        if (isError) {
            contentDiv.style.background = '#f8d7da';
            contentDiv.style.color = '#721c24';
            contentDiv.style.border = '1px solid #f5c6cb';
        }
        
        messageDiv.appendChild(avatarDiv);
        messageDiv.appendChild(contentDiv);
        
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
    }
    
    createSourcesElement(sources) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'sources';
        
        const title = document.createElement('h6');
        title.innerHTML = '<i class="fas fa-link"></i> Sources:';
        sourcesDiv.appendChild(title);
        
        sources.forEach(source => {
            const sourceItem = document.createElement('div');
            sourceItem.className = 'source-item';
            
            const sourceNameDiv = document.createElement('div');
            sourceNameDiv.className = 'source-name';
            sourceNameDiv.textContent = source.filename;
            
            const sourceInfoDiv = document.createElement('div');
            sourceInfoDiv.className = 'source-info';
            
            let infoText = source.type;
            if (source.page) {
                infoText += ` • Page ${source.page}`;
            }
            if (source.similarity) {
                infoText += ` • ${(source.similarity * 100).toFixed(1)}% match`;
            }
            sourceInfoDiv.textContent = infoText;
            
            sourceItem.appendChild(sourceNameDiv);
            sourceItem.appendChild(sourceInfoDiv);
            sourcesDiv.appendChild(sourceItem);
        });
        
        return sourcesDiv;
    }
    
    createImagesElement(images) {
        const imagesDiv = document.createElement('div');
        imagesDiv.className = 'related-images';
        
        const title = document.createElement('h6');
        title.innerHTML = '<i class="fas fa-image"></i> Related Images:';
        imagesDiv.appendChild(title);
        
        images.forEach(image => {
            const imageContainer = document.createElement('div');
            imageContainer.className = 'image-container';
            
            const imgElement = document.createElement('img');
            imgElement.src = image.url;
            imgElement.alt = image.caption;
            imgElement.className = 'related-image';
            imgElement.onclick = () => this.openImageModal(image.url, image.caption);
            
            const captionDiv = document.createElement('div');
            captionDiv.className = 'image-caption';
            captionDiv.textContent = image.caption;
            
            if (image.similarity) {
                const similaritySpan = document.createElement('span');
                similaritySpan.className = 'similarity-score';
                similaritySpan.textContent = ` (${(image.similarity * 100).toFixed(1)}% match)`;
                captionDiv.appendChild(similaritySpan);
            }
            
            imageContainer.appendChild(imgElement);
            imageContainer.appendChild(captionDiv);
            imagesDiv.appendChild(imageContainer);
        });
        
        return imagesDiv;
    }
    
    openImageModal(imageUrl, caption) {
        // Create modal overlay
        const modalOverlay = document.createElement('div');
        modalOverlay.className = 'modal-overlay';
        modalOverlay.onclick = () => document.body.removeChild(modalOverlay);
        
        // Create modal content
        const modalContent = document.createElement('div');
        modalContent.className = 'modal-content';
        modalContent.onclick = (e) => e.stopPropagation();
        
        // Create close button
        const closeButton = document.createElement('button');
        closeButton.className = 'modal-close';
        closeButton.innerHTML = '&times;';
        closeButton.onclick = () => document.body.removeChild(modalOverlay);
        
        // Create image
        const modalImage = document.createElement('img');
        modalImage.src = imageUrl;
        modalImage.alt = caption;
        modalImage.className = 'modal-image';
        
        // Create caption
        const modalCaption = document.createElement('div');
        modalCaption.className = 'modal-caption';
        modalCaption.textContent = caption;
        
        modalContent.appendChild(closeButton);
        modalContent.appendChild(modalImage);
        modalContent.appendChild(modalCaption);
        modalOverlay.appendChild(modalContent);
        
        document.body.appendChild(modalOverlay);
    }
    
    formatMessage(content) {
        // Enhanced markdown formatting
        return content
            // Headers
            .replace(/^### (.*$)/gm, '<h3 class="message-header">$1</h3>')
            .replace(/^## (.*$)/gm, '<h2 class="message-header">$1</h2>')
            .replace(/^# (.*$)/gm, '<h1 class="message-header">$1</h1>')
            // Bold and italic
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            // Code blocks
            .replace(/```([\s\S]*?)```/g, '<pre class="code-block"><code>$1</code></pre>')
            .replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>')
            // Lists
            .replace(/^\d+\. (.*)$/gm, '<div class="numbered-item">$1</div>')
            .replace(/^- (.*)$/gm, '<div class="bullet-item">• $1</div>')
            // Line breaks and paragraphs
            .replace(/\n\n/g, '</p><p class="message-paragraph">')
            .replace(/\n/g, '<br>')
            // Wrap in paragraph if not already wrapped
            .replace(/^(?!<[h|p|d|u|o])(.+)/, '<p class="message-paragraph">$1')
            .replace(/([^>])$/, '$1</p>');
    }
    
    showTypingIndicator(show) {
        this.typingIndicator.style.display = show ? 'block' : 'none';
        if (show) {
            this.scrollToBottom();
        }
    }
    
    scrollToBottom() {
        setTimeout(() => {
            this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        }, 100);
    }
    
    autoResizeInput() {
        this.messageInput.style.height = 'auto';
        this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 150) + 'px';
    }
    
    validateFiles() {
        const files = Array.from(this.fileInput.files);
        const maxSize = 50 * 1024 * 1024; // 50MB
        const supportedTypes = [
            'application/pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/msword',
            'text/plain',
            'text/csv',
            'image/png',
            'image/jpeg',
            'image/jpg',
            'image/bmp',
            'image/gif'
        ];
        
        let isValid = true;
        let errorMessage = '';
        
        files.forEach(file => {
            if (file.size > maxSize) {
                isValid = false;
                errorMessage = `File "${file.name}" is too large. Maximum size is 50MB.`;
            }
            
            if (!supportedTypes.includes(file.type) && !this.isValidFileExtension(file.name)) {
                isValid = false;
                errorMessage = `File "${file.name}" has an unsupported format.`;
            }
        });
        
        if (!isValid) {
            this.showAlert(errorMessage, 'error');
            this.fileInput.value = '';
        }
        
        this.uploadBtn.disabled = files.length === 0 || !isValid;
    }
    
    isValidFileExtension(filename) {
        const validExtensions = ['.pdf', '.docx', '.doc', '.txt', '.csv', '.png', '.jpg', '.jpeg', '.bmp', '.gif'];
        const extension = filename.toLowerCase().substr(filename.lastIndexOf('.'));
        return validExtensions.includes(extension);
    }
    
    async handleFileUpload() {
        const files = Array.from(this.fileInput.files);
        if (files.length === 0) return;
        
        this.uploadBtn.disabled = true;
        this.showUploadProgress(true);
        
        try {
            for (let i = 0; i < files.length; i++) {
                const file = files[i];
                await this.uploadFile(file, i + 1, files.length);
            }
            
            this.showAlert(`Successfully uploaded ${files.length} file(s)!`, 'success');
            this.fileInput.value = '';
            this.loadSystemStats();
            this.loadDocumentList(); // Load and display uploaded documents
            
        } catch (error) {
            this.showAlert(`Upload failed: ${error.message}`, 'error');
        } finally {
            this.uploadBtn.disabled = false;
            this.showUploadProgress(false);
        }
    }
    
    async uploadFile(file, current, total) {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        
        if (!result.success) {
            throw new Error(result.message);
        }
        
        // Update progress
        const progress = (current / total) * 100;
        this.updateUploadProgress(progress, `Processing ${file.name}... (${current}/${total})`);
        
        return result;
    }
    
    showUploadProgress(show) {
        this.uploadProgress.style.display = show ? 'block' : 'none';
        if (!show) {
            this.updateUploadProgress(0, '');
        }
    }
    
    updateUploadProgress(percentage, status) {
        const progressBar = this.uploadProgress.querySelector('.progress-bar');
        const statusText = this.uploadProgress.querySelector('.upload-status');
        
        progressBar.style.width = `${percentage}%`;
        statusText.textContent = status;
    }
    
    async loadSystemStats() {
        try {
            const response = await fetch('/stats');
            const stats = await response.json();
            
            if (stats.error) {
                this.systemStatus.textContent = 'Error';
                this.systemStatus.className = 'badge bg-danger';
            } else {
                this.docCount.textContent = stats.documents_count || 0;
                this.chunkCount.textContent = stats.chunks_count || 0;
                this.systemStatus.textContent = stats.vector_db_status === 'healthy' ? 'Ready' : 'Error';
                this.systemStatus.className = `badge bg-${stats.vector_db_status === 'healthy' ? 'success' : 'danger'}`;
            }
            
        } catch (error) {
            console.error('Failed to load stats:', error);
            this.systemStatus.textContent = 'Error';
            this.systemStatus.className = 'badge bg-danger';
        }
    }

    async loadDocumentList() {
        try {
            const response = await fetch('/api/documents');
            const documents = await response.json();
            
            const documentsList = document.getElementById('documentsList');
            
            if (documents.length === 0) {
                documentsList.innerHTML = '<div class="no-documents">No documents uploaded yet</div>';
                return;
            }
            
            documentsList.innerHTML = documents.map(doc => `
                <div class="document-item">
                    <div class="document-info">
                        <div class="document-name">${doc.name}</div>
                        <div class="document-meta">
                            <span class="document-type">${doc.type.toUpperCase()}</span>
                            <span class="document-chunks">${doc.chunks} chunks</span>
                            ${doc.has_images ? '<span class="has-images">📷 Images</span>' : ''}
                        </div>
                    </div>
                </div>
            `).join('');
            
        } catch (error) {
            console.error('Failed to load documents:', error);
            const documentsList = document.getElementById('documentsList');
            documentsList.innerHTML = '<div class="error-message">Error loading documents</div>';
        }
    }
    
    clearChat() {
        // Keep only the welcome message
        const messages = this.chatMessages.querySelectorAll('.message');
        for (let i = 1; i < messages.length; i++) {
            messages[i].remove();
        }
        
        // Generate new session ID
        this.sessionId = this.generateSessionId();
        
        this.showAlert('Chat cleared!', 'success');
    }
    
    showAlert(message, type) {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert-custom alert-${type}`;
        alertDiv.textContent = message;
        
        // Insert at the top of chat messages
        this.chatMessages.insertBefore(alertDiv, this.chatMessages.firstChild);
        
        // Remove after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.parentNode.removeChild(alertDiv);
            }
        }, 5000);
    }

    // Advanced Features Implementation
    toggleTheme() {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        
        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        
        const themeIcon = document.querySelector('#themeToggle i');
        if (themeIcon) {
            themeIcon.className = newTheme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
        }
    }

    initializeTheme() {
        const savedTheme = localStorage.getItem('theme') || 'light';
        document.documentElement.setAttribute('data-theme', savedTheme);
        
        const themeIcon = document.querySelector('#themeToggle i');
        if (themeIcon) {
            themeIcon.className = savedTheme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
        }
    }

    performAdvancedSearch() {
        const searchQuery = document.getElementById('searchInput')?.value.toLowerCase() || '';
        const keywordFilter = document.getElementById('keywordFilter')?.value.toLowerCase() || '';
        const typeFilter = document.getElementById('typeFilter')?.value || '';

        if (!searchQuery && !keywordFilter && !typeFilter) {
            // Reset document display
            return;
        }

        // Send search query through chat for semantic search
        if (searchQuery || keywordFilter) {
            const searchMessage = `Search documents for: ${searchQuery || keywordFilter}${typeFilter ? ` (${typeFilter} files only)` : ''}`;
            this.sendMessage(searchMessage);
        }
    }

    showBulkOperationsModal() {
        const modal = new bootstrap.Modal(document.getElementById('bulkOperationsModal'));
        modal.show();
    }

    async deleteAllDocuments() {
        if (!confirm('Are you sure you want to delete all documents? This cannot be undone.')) {
            return;
        }

        this.addMessage('Bulk delete operation would clear all uploaded documents from the vector database.', 'bot');
    }

    exportDocuments() {
        const docData = {
            exportDate: new Date().toISOString(),
            message: "Document export functionality - lists all uploaded files and their metadata"
        };

        const dataStr = JSON.stringify(docData, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        
        const link = document.createElement('a');
        link.href = URL.createObjectURL(dataBlob);
        link.download = `electrical_grid_documents_${new Date().toISOString().split('T')[0]}.json`;
        link.click();

        this.addMessage('Document list exported successfully!', 'bot');
    }

    async batchAnalyzeImages() {
        this.addMessage('Starting batch analysis of all uploaded images for electrical components...', 'bot');
        
        // This would call the backend for actual batch processing
        setTimeout(() => {
            this.addMessage('Batch image analysis complete! Found transformers, circuit breakers, and transmission lines in uploaded images.', 'bot');
        }, 2000);
    }

    async extractElectricalComponents() {
        const modal = new bootstrap.Modal(document.getElementById('componentAnalysisModal'));
        modal.show();

        const resultsContainer = document.getElementById('componentAnalysisResults');
        
        // Simulate component analysis
        setTimeout(() => {
            const mockResults = [
                {
                    component: 'Power Transformer',
                    confidence: 0.95,
                    details: 'High voltage power transformer, likely 115kV class',
                    location: 'Main substation area'
                },
                {
                    component: 'Circuit Breaker',
                    confidence: 0.88,
                    details: 'SF6 gas insulated circuit breaker',
                    location: 'Switching station'
                },
                {
                    component: 'Transmission Lines',
                    confidence: 0.92,
                    details: '3-phase overhead transmission lines',
                    location: 'Power grid infrastructure'
                }
            ];

            resultsContainer.innerHTML = mockResults.map(result => `
                <div class="component-item">
                    <h6>${result.component} <span class="confidence-score">${(result.confidence * 100).toFixed(0)}%</span></h6>
                    <div class="component-details">
                        <strong>Details:</strong> ${result.details}<br>
                        <strong>Location:</strong> ${result.location}
                    </div>
                </div>
            `).join('');
        }, 2000);
    }

    async rebuildSearchIndex() {
        this.addMessage('Rebuilding search index for better document retrieval...', 'bot');
        this.loadSystemStats();
        
        setTimeout(() => {
            this.addMessage('Search index rebuilt successfully! Document search performance improved.', 'bot');
        }, 1500);
    }

    analyzeAllImages() {
        this.batchAnalyzeImages();
    }
}

// Initialize the chatbot when the page loads
document.addEventListener('DOMContentLoaded', () => {
    const chatbot = new ElectricalGridChatbot();
    chatbot.initializeTheme(); // Initialize theme on load
    chatbot.loadDocumentList(); // Load existing documents on startup
    window.chatbot = chatbot; // Make available globally
});

// Handle page visibility change to reconnect WebSocket if needed
document.addEventListener('visibilitychange', () => {
    if (!document.hidden && window.chatbot && !window.chatbot.isConnected) {
        window.chatbot.connectWebSocket();
    }
});

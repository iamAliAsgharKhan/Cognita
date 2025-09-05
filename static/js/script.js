document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const chatWindow = document.getElementById('chat-window');
    const chatForm = document.getElementById('chat-form');
    const queryInput = document.getElementById('query');
    const submitButton = document.getElementById('submit-button');
    const typingIndicator = document.getElementById('typing-indicator');
    
    // Sidebar Elements
    const historySidebar = document.getElementById('history-sidebar');
    const toggleHistoryBtn = document.getElementById('toggleHistoryBtn');

    // Settings Modal Elements
    const openSettingsBtn = document.getElementById('openSettingsBtn');
    const closeSettingsBtn = document.getElementById('closeSettingsBtn');
    const settingsModalBackdrop = document.getElementById('settings-modal-backdrop');
    const directoryInput = document.getElementById('directoryPath');
    const saveDirectoryBtn = document.getElementById('saveDirectoryBtn');
    const browseDirectoryBtn = document.getElementById('browseDirectoryBtn');
    const modelSelect = document.getElementById('modelName');
    const reindexBtn = document.getElementById('reindexBtn');

    // --- 1. Add Messages to Chat Window ---
    const addMessage = (data, sender) => {
        const messageElement = document.createElement('div');
        messageElement.classList.add('chat-message', sender);

        if (sender === 'user' || sender === 'system') {
            messageElement.innerHTML = marked.parse(data);
        } 
        else if (sender === 'bot') {
            const responseText = (typeof data === 'object' && data.response) ? data.response : data.toString();
            const sources = (typeof data === 'object' && data.sources) ? data.sources : [];

            const responseContent = document.createElement('div');
            responseContent.classList.add('response-content');
            responseContent.innerHTML = marked.parse(responseText);
            messageElement.appendChild(responseContent);

            if (sources.length > 0) {
                const sourcesContainer = document.createElement('div');
                sourcesContainer.classList.add('sources-container');
                const sourcesLabel = document.createElement('span');
                sourcesLabel.classList.add('sources-label');
                sourcesLabel.textContent = 'Sources:';
                sourcesContainer.appendChild(sourcesLabel);
                sources.forEach(sourceFile => {
                    const sourceItem = document.createElement('span');
                    sourceItem.classList.add('source-item');
                    sourceItem.textContent = sourceFile;
                    sourcesContainer.appendChild(sourceItem);
                });
                messageElement.appendChild(sourcesContainer);
            }
        }

        chatWindow.appendChild(messageElement);
        chatWindow.scrollTop = chatWindow.scrollHeight;
    };

    // --- 2. Initial Data Loading ---
    const fetchModels = async () => {
        try {
            const response = await fetch('/models');
            if (!response.ok) throw new Error('Failed to fetch models');
            const data = await response.json();
            modelSelect.innerHTML = '';
            data.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = model;
                modelSelect.appendChild(option);
            });
        } catch (error) {
            addMessage(`**Error:** Could not load embedding models. ${error.message}`, 'system');
        }
    };

    const fetchDirectory = async () => {
        try {
            const response = await fetch('/directory');
            if (!response.ok) throw new Error('Failed to fetch directory');
            const data = await response.json();
            directoryInput.value = data.path;
        } catch (error) {
            addMessage(`**Error:** Could not load current directory. ${error.message}`, 'system');
        }
    };

    // --- 3. UI State Management ---
    const showTypingIndicator = () => { typingIndicator.style.display = 'flex'; chatWindow.scrollTop = chatWindow.scrollHeight; };
    const hideTypingIndicator = () => { typingIndicator.style.display = 'none'; };

    const setUIBusy = (isBusy) => {
        submitButton.disabled = isBusy;
        queryInput.disabled = isBusy;
    };
    
    const setSettingsBusy = (isBusy) => {
        saveDirectoryBtn.disabled = isBusy;
        reindexBtn.disabled = isBusy;
        browseDirectoryBtn.disabled = isBusy;
        openSettingsBtn.disabled = isBusy;
    }

    // --- 4. Event Listeners ---
    // Sidebar Toggle
    toggleHistoryBtn.addEventListener('click', () => {
        historySidebar.classList.toggle('collapsed');
        // Save the state to localStorage
        const isCollapsed = historySidebar.classList.contains('collapsed');
        localStorage.setItem('sidebarState', isCollapsed ? 'collapsed' : 'expanded');
    });

    const showSettingsModal = () => {
        settingsModalBackdrop.style.display = 'flex';
        setTimeout(() => settingsModalBackdrop.classList.add('visible'), 10);
    };

    const hideSettingsModal = () => {
        settingsModalBackdrop.classList.remove('visible');
        setTimeout(() => settingsModalBackdrop.style.display = 'none', 300); // Match CSS transition duration
    };

    openSettingsBtn.addEventListener('click', showSettingsModal);
    closeSettingsBtn.addEventListener('click', hideSettingsModal);
    settingsModalBackdrop.addEventListener('click', (event) => {
        if (event.target === settingsModalBackdrop) {
            hideSettingsModal();
        }
    });

    browseDirectoryBtn.addEventListener('click', () => {
        alert("Browser Security Limitation\n\nPlease open your file explorer, navigate to your notes directory, copy the full path from the address bar, and then paste it into the input field.");
        directoryInput.focus();
    });

    saveDirectoryBtn.addEventListener('click', async () => {
        const newPath = directoryInput.value.trim();
        if (!newPath) {
            addMessage('**Error:** Directory path cannot be empty.', 'system');
            return;
        }
        addMessage(`_Attempting to set new directory to "${newPath}"..._`, 'system');
        setSettingsBusy(true);
        try {
            const response = await fetch('/directory', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path: newPath }),
            });
            const result = await response.json();
            if (!response.ok) throw new Error(result.detail);
            addMessage(`**Success!** ${result.message}`, 'system');
            hideSettingsModal();
        } catch (error) {
            addMessage(`**Failed to set directory:** ${error.message}`, 'system');
        } finally {
            setSettingsBusy(false);
        }
    });

    reindexBtn.addEventListener('click', async () => {
        const selectedModel = modelSelect.value;
        addMessage(`_Starting re-indexing with model **${selectedModel}**... This may take a moment._`, 'system');
        setSettingsBusy(true);
        hideSettingsModal();
        try {
            const response = await fetch('/reindex', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_name: selectedModel }),
            });
            const result = await response.json();
            if (!response.ok) throw new Error(result.detail);
            addMessage(`**Success!** ${result.message}`, 'system');
        } catch (error) {
            addMessage(`**Re-indexing failed:** ${error.message}`, 'system');
        } finally {
            setSettingsBusy(false);
        }
    });

    chatForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        const query = queryInput.value.trim();
        if (!query) return;
        addMessage(query, 'user');
        queryInput.value = '';
        setUIBusy(true);
        showTypingIndicator();
        try {
            const response = await fetch('/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: query,
                    model_name: modelSelect.value,
                    max_tokens: 6000
                }),
            });
            const result = await response.json();
            if (!response.ok) throw new Error(result.detail);
            addMessage(result, 'bot');
        } catch (error) {
            addMessage(`**Sorry, an error occurred:** ${error.message}`, 'bot');
        } finally {
            setUIBusy(false);
            hideTypingIndicator();
            queryInput.focus();
        }
    });

    // --- Initial Application Load ---
    const initializeApp = async () => {
        // Restore sidebar state from localStorage. The default is collapsed in the HTML.
        // This will only expand it if the user explicitly left it open in a previous session.
        const sidebarState = localStorage.getItem('sidebarState');
        if (sidebarState === 'expanded') {
            historySidebar.classList.remove('collapsed');
        }

        addMessage("Hello! I'm Cognita, ready to answer questions about your Obsidian notes. What would you like to know?", 'bot');
        await fetchModels();
        await fetchDirectory();
        try {
            const response = await fetch('/status');
            if (!response.ok) throw new Error("Could not fetch server status.");
            const data = await response.json();
            const collections = data.indexed_collections || {};
            const totalDocs = Object.values(collections).reduce((sum, count) => sum + count, 0);
            if (totalDocs === 0) {
                addMessage(
                    "**Notice:** No documents are currently indexed. Please open the settings (⚙️) to verify your directory path and use the re-index button.",
                    'system'
                );
            }
        } catch (error) {
            console.error("Could not fetch status on startup:", error);
            addMessage(`**Warning:** Could not verify index status. ${error.message}`, 'system');
        }
    };

    initializeApp();
});


// Changesomething to test
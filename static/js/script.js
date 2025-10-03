/**
 * Adiciona os listeners e a lógica principal do chat quando o DOM está totalmente carregado.
 */
document.addEventListener('DOMContentLoaded', () => {
    // --- Seletores de Elementos do DOM ---
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatBox = document.getElementById('chat-box');
    
    // Flag para evitar envios múltiplos enquanto uma resposta está sendo processada.
    let isSending = false;

    /**
     * Listener para o envio do formulário (clique no botão ou Enter).
     */
    chatForm.addEventListener('submit', (event) => {
        event.preventDefault();
        
        // Impede o envio se uma requisição já estiver em andamento.
        if (isSending) return;

        const message = userInput.value.trim();
        if (message) {
            userInput.value = '';
            sendMessageFlow(message);
        }
    });

    /**
     * Orquestra todo o fluxo de envio de mensagem e recebimento de resposta.
     * @param {string} message - A mensagem digitada pelo usuário.
     */
    async function sendMessageFlow(message) {
        // 1. Adiciona a mensagem do usuário à tela.
        addMessageToChatBox('user', message);
        isSending = true;

        // 2. Mostra o indicador de "digitando".
        const typingIndicator = createTypingIndicator();

        try {
            // 3. Envia a mensagem para a API.
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message }),
            });

            // 4. Se a resposta da API não for bem-sucedida, trata o erro.
            if (!response.ok) {
                const errorData = await response.json();
                // Usa a mensagem de erro da API se disponível, senão uma genérica.
                throw new Error(errorData.error || `Erro na API: ${response.statusText}`);
            }

            const data = await response.json();

            // 5. Remove o indicador de "digitando" antes de mostrar as respostas.
            typingIndicator.remove();

            // 6. Se não houver respostas, exibe uma mensagem padrão.
            if (!data.responses || data.responses.length === 0) {
                addMessageToChatBox('bot', 'Não entendi muito bem. Você pode tentar perguntar sobre nosso cardápio, promoções ou tempo de entrega?');
                return;
            }

            // 7. Exibe cada resposta do bot com um pequeno atraso para parecer mais natural.
            for (const res of data.responses) {
                const probabilityPercent = Math.round(res.probability * 100);
                const metaText = `Intenção: ${res.intent} · Confiança: ${probabilityPercent}%`;
                addMessageToChatBox('bot', res.answer, metaText);
                await delay(500); // Espera 500ms entre cada mensagem.
            }

        } catch (error) {
            console.error('Falha ao comunicar com o chatbot:', error);
            // Remove o indicador de "digitando" também em caso de erro.
            typingIndicator.remove();
            addMessageToChatBox('bot', `Ops! Tive um problema para me conectar. Erro: ${error.message}`);
        } finally {
            // 8. Reabilita o envio e foca no campo de input.
            isSending = false;
            userInput.focus();
        }
    }
    
    /**
     * Adiciona uma nova mensagem (usuário ou bot) à caixa de chat.
     * @param {'user' | 'bot'} sender - Quem enviou a mensagem.
     * @param {string} message - O conteúdo da mensagem.
     * @param {string} [metaText] - Texto opcional com metadados (intenção, confiança).
     */
    function addMessageToChatBox(sender, message, metaText) {
        const wrapper = document.createElement('div');
        wrapper.classList.add('message', `message--${sender}`);

        const avatar = sender === 'bot' ? createAvatar('🍟') : '';
        const metaHTML = metaText ? `<div class="bubble__meta">${metaText}</div>` : '';

        // Usa template literals para criar o HTML de forma mais limpa.
        wrapper.innerHTML = `
            ${avatar}
            <div class="bubble">
                ${message}
                ${metaHTML}
            </div>
        `;
        
        chatBox.appendChild(wrapper);
        scrollToEnd();
    }

    /**
     * Cria e exibe o indicador visual de "digitando".
     * @returns {HTMLElement} O elemento do indicador para que possa ser removido depois.
     */
    function createTypingIndicator() {
        const wrapper = document.createElement('div');
        wrapper.classList.add('message', 'message--bot', 'message--typing');
        wrapper.innerHTML = `
            ${createAvatar('🍟')}
            <div class="bubble">
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
            </div>
        `;
        chatBox.appendChild(wrapper);
        scrollToEnd();
        return wrapper;
    }

    /**
     * Cria o HTML do avatar do bot.
     * @param {string} symbol - O emoji ou símbolo para o avatar.
     * @returns {string} A string HTML do avatar.
     */
    function createAvatar(symbol) {
        return `<div class="avatar">${symbol}</div>`;
    }

    /**
     * Rola a caixa de chat para o final para mostrar a mensagem mais recente.
     */
    function scrollToEnd() {
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    /**
     * Uma função utilitária para criar um atraso (delay).
     * @param {number} ms - O tempo de atraso em milissegundos.
     * @returns {Promise<void>}
     */
    const delay = ms => new Promise(res => setTimeout(res, ms));
});


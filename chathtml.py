css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://www.state.gov/wp-content/uploads/2023/01/PEPFAR-20-Logo-Social-Tagged.jpg">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://banner2.cleanpng.com/20180820/vzw/kisspng-portable-network-graphics-clip-art-question-mark-c-the-unanswerable-questions-orchard-hill-church-5b7b5cb3aecd99.240854201534811315716.jpg">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
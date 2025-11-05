import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime
import base64
from config import Config

class NotificationSystem:
    """Система уведомлений о подозрительных событиях"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def send_email_alert(self, event_data, frame_data=None):
        """Отправляет email уведомление о подозрительном событии"""
        if not self.config['ENABLE_EMAIL_ALERTS']:
            return False
            
        try:
            # Создаем сообщение
            msg = MIMEMultipart()
            msg['From'] = self.config['EMAIL_USER']
            msg['To'] = self.config['EMAIL_USER']  # Можно настроить получателя
            event_type = event_data.get('event_type') or event_data.get('type') or 'unknown_event'
            msg['Subject'] = f"🚨 СИСТЕМА БЕЗОПАСНОСТИ: {event_type}"
            
            # Текст сообщения
            body = f"""
            <html>
            <body>
                <h2>🚨 Обнаружено подозрительное событие!</h2>
                <p><strong>Время:</strong> {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}</p>
                <p><strong>Тип события:</strong> {event_type}</p>
                <p><strong>Описание:</strong> {event_data['description']}</p>
                <p><strong>Уверенность:</strong> {float(event_data.get('confidence', 0.0)):.2%}</p>
                <p><strong>Местоположение:</strong> {event_data.get('location', 'Не указано')}</p>
                
                <hr>
                <p><em>Это автоматическое уведомление от системы анализа видеопотока.</em></p>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(body, 'html'))
            
            # Добавляем изображение, если есть
            if frame_data:
                try:
                    frame_bytes = base64.b64decode(frame_data)
                    image = MIMEImage(frame_bytes)
                    image.add_header('Content-Disposition', 'attachment', filename='suspicious_event.jpg')
                    msg.attach(image)
                except Exception as e:
                    self.logger.warning(f"Не удалось прикрепить изображение: {e}")
            
            # Отправляем email
            server = smtplib.SMTP(self.config['SMTP_SERVER'], self.config['SMTP_PORT'])
            server.starttls()
            server.login(self.config['EMAIL_USER'], self.config['EMAIL_PASSWORD'])
            text = msg.as_string()
            server.sendmail(self.config['EMAIL_USER'], self.config['EMAIL_USER'], text)
            server.quit()
            
            self.logger.info(f"Email уведомление отправлено для события: {event_type}")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при отправке email: {e}")
            return False
    
    def log_event(self, event_data):
        """Логирует событие в файл"""
        try:
            event_type = event_data.get('event_type') or event_data.get('type') or 'unknown_event'
            log_message = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] " \
                         f"СОБЫТИЕ: {event_type} - {event_data.get('description','')} " \
                         f"(уверенность: {float(event_data.get('confidence',0.0)):.2%})"
            
            self.logger.info(log_message)
            
        except Exception as e:
            self.logger.error(f"Ошибка при логировании события: {e}")
    
    def send_alert(self, event_data, frame_data=None):
        """Отправляет уведомление всеми доступными способами"""
        # Логируем событие
        self.log_event(event_data)
        
        # Отправляем email, если включен
        if self.config['ENABLE_EMAIL_ALERTS']:
            self.send_email_alert(event_data, frame_data)
        
        return True

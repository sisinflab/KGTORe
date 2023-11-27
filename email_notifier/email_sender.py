import os.path
import socket
import yaml
import smtplib, ssl
from email.message import EmailMessage


class EmailNotifier:
    def __init__(self, configuration_path=None):
        if configuration_path is None:
            configuration_path = os.path.join(os.path.dirname(__file__), 'config.yml')

        self.senders_field = 'senders'
        self.receivers_field = 'receivers'
        self.messages_field = 'messages'
        self.server_field = 'server'
        self.port_field = 'port'

        self.ok_message_field = 'ok'
        self.error_message_field = 'error'
        self.default_ok_subject = 'esperimento terminato'
        self.default_error_subject = 'si è verificato un errore durante l\'esperimento'
        self.default_ok_message = 'job terminato'
        self.default_error_message = 'job in errore'

        self.default_server = 'smtp.gmail.com'
        self.default_port = 465

        self.senders, self.receivers, self.messages, self.server, self.port = \
            self.read_configuration_file(configuration_path)

    def read_configuration_file(self, path) -> tuple:
        """
        Given the path of the configuration file returns the parameters necessary for the class.
        If a not mandatory field is missing the value is replaced with the default value.
        Mandatory fields: 'senders' and 'receivers'
        @param path: path of the configuration file
        @return: the senders, the receivers, the messages, the server address and the port
        """

        print(f'Reading configuration file from \'{path}\'')
        with open(path, encoding='utf8') as file:
            config = yaml.safe_load(file)
        mandatory_fields = [self.senders_field, self.receivers_field]
        for field in mandatory_fields:
            if field not in config:
                raise AttributeError(f'Field {field} is missing in the configuration file. Please, fix it.')
        senders = config[self.senders_field]
        receivers = config[self.receivers_field]

        if self.messages_field in config:
            messages = config[self.messages_field]
        else:
            messages = [{'role': self.ok_message_field,
                         'subject': self.default_ok_subject,
                         'body': self.default_ok_message},
                        {'role': self.error_message_field,
                         'subject': self.default_error_subject,
                         'body': self.default_error_message}]

        if self.server_field in config:
            server = config[self.server_field]
        else:
            server = self.default_server

        if self.port_field in config:
            port = config[self.port_field]
        else:
            port = self.default_port

        return senders, receivers, messages, server, port

    def send_ok(self, additional_body=None):
        """
        Send a success email
        @param additional_body: a message that is appended to the body
        @return: None
        """
        message = dict(self.messages[self.ok_message_field])
        if additional_body:
            message['body'] += '\n' + additional_body
        for sender in self.senders:
            sender_mail, sender_pass = sender['mail'], sender['pass']
            for receiver in self.receivers:
                self.send(sender_mail, receiver, message, sender_pass)

    def send_error(self, additional_body=None, exception=None):
        """
        Send an error email
        @param additional_body: a message that is appended to the body
        @param exception: error exception class
        @return: None
        """

        message = dict(self.messages[self.error_message_field])
        if additional_body:
            message['body'] += '\n' + additional_body
        if exception:
            message['body'] += '\n' + f'Error type: {exception}'

        for sender in self.senders:
            sender_mail = sender['mail']
            sender_pass = sender['pass']
            for receiver in self.receivers:
                self.send(sender_mail, receiver, message, sender_pass)

    def send(self, sender, receiver, message, password, smtp_server=None, port=None):
        """
        Send an email
        @param sender: email sender
        @param receiver: email receiver
        @param message: email message
        @param password: sender password
        @param smtp_server: smtp server address
        @param port: port
        @return: None
        """
        if smtp_server is None:
            smtp_server = self.server
        if port is None:
            port = self.port

        print(f'sending message from {sender} to {receiver}')
        context = ssl.create_default_context()

        message_obj = EmailMessage()
        message_obj['Subject'] = f"{message['subject']} ({socket.gethostname()})"
        message_obj['From'] = sender
        message_obj['To'] = receiver
        message_obj.set_content(message['body'])

        try:
            with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
                server.login(sender, password)
                server.send_message(message_obj)
            print(f'email from {sender} to {receiver} sent')
        except ConnectionError:
            print(f'an error occurred while sending an email from {sender} to {receiver} sent')

    def notify(self, func, *args, additional_body=None, **kwargs):
        """
        Wrapper for sending an email both if the function succeeds or fails.
        It is possible to add a message to the body for both cases.
        @param func: the called function
        @param args: the positional arguments of the called function
        @param additional_body: the text that will be attached to the email body
        @param kwargs: keyword arguments of the called function
        @return: the result of the function or None if fails
        """
        try:
            func(*args, **kwargs)
            self.send_ok(additional_body)
        except Exception as e:
            self.send_error(additional_body=additional_body, exception=e)



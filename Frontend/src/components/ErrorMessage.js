import React from 'react';
import '../App.css'; 

const ErrorMessage = ({ message, onClose }) => (
    <div className="error-message">
        <span>{message}</span>
        {onClose && <button onClick={onClose} className="close-error-btn">✖</button>}
    </div>
);

export default ErrorMessage;
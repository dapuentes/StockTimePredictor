import axios from 'axios';

// URL del API Gateway
const API_GATEWAY_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

/**
 * Trains a machine learning model using the specified configuration and model type.
 *
 * @async
 * @function trainModel
 * @param {string} modelType - The type of model to train (e.g., "linear_regression", "neural_network").
 * @param {Object} config - Configuration object for the training process.
 * @param {string} config.selectedTicker - The ticker symbol of the stock or asset to train on.
 * @param {Date} config.startDate - The start date for the training data.
 * @param {Date} config.endDate - The end date for the training data.
 * @param {number} [config.nLags=10] - The number of lag features to include in the training data.
 * @param {string} [config.targetCol='Close'] - The target column to predict (e.g., "Close", "Open").
 * @param {number} [config.trainSize=0.8] - The proportion of data to use for training (between 0 and 1).
 * @throws {Error} Throws an error if required configuration fields are missing or if the training request fails.
 * @returns {Promise<Object>} A promise that resolves to the response data from the training API,
 * containing details such as metrics, best parameters, and the model path.
 */
export const trainModel = async (modelType, config) => {
    // Extraer y validar configuración
    const {
        selectedTicker,
        startDate,
        endDate,
        nLags = 10,
        targetCol = 'Close',
        trainSize = 0.8
    } = config;

    // Validar que las fechas y el ticker no sean nulos o indefinidos
    if (!selectedTicker || !startDate || !endDate) {
        const errorMessage = "Ticker, fecha de inicio o fecha de fin faltantes en la configuración.";
        console.error("Error en trainModel:", errorMessage);
        throw new Error(errorMessage);
    }

    // Formatear fechas a YYYY-MM-DD
    const startStr = startDate.toISOString().split('T')[0];
    const endStr = endDate.toISOString().split('T')[0];

    // Crear objeto FormData para enviar los datos
    const formData = new FormData();
    formData.append('ticket', selectedTicker);
    formData.append('start_date', startStr);
    formData.append('end_date', endStr);
    formData.append('n_lags', String(nLags)); // Convertir número a string
    formData.append('target_col', targetCol);
    formData.append('train_size', String(trainSize)); // Convertir número a string

    // Construir la URL del endpoint
    const url = `${API_GATEWAY_URL}/train/${modelType}`;

    try {
        console.log(`Enviando solicitud de entrenamiento a: ${url}`);
        // Es útil loggear los datos que se van a enviar, aunque FormData no se loggea tan directamente
        console.log("Datos a enviar (FormData):", {
             ticket: selectedTicker,
             start_date: startStr,
             end_date: endStr,
             n_lags: String(nLags),
             target_col: targetCol,
             train_size: String(trainSize)
         });

        // Enviar la solicitud POST con FormData
        const response = await axios.post(url, formData);

        console.log("Respuesta del entrenamiento:", response.data);
        return response.data;

    } catch (error) {
        // Mejor manejo de errores
        let errorMessage = "Error desconocido entrenando el modelo.";
        if (error.response) {
            // El servidor respondió con un código de estado fuera del rango 2xx
            console.error("Error Response Data:", error.response.data);
            console.error("Error Response Status:", error.response.status);
            console.error("Error Response Headers:", error.response.headers);
            // Intenta extraer un mensaje más específico si el backend lo envía
            errorMessage = typeof error.response.data === 'object' ? JSON.stringify(error.response.data.detail || error.response.data) : String(error.response.data);
        } else if (error.request) {
            // La solicitud se hizo pero no se recibió respuesta
            console.error("Error Request:", error.request);
            errorMessage = "No se recibió respuesta del servidor. Verifica que el API Gateway esté corriendo.";
        } else {
            // Algo pasó al configurar la solicitud
            console.error('Error Message:', error.message);
            errorMessage = error.message;
        }
        // Lanza un error que pueda ser capturado por el componente que llamó a esta función
        throw new Error(`Error en el entrenamiento: ${errorMessage}`);
    }
};


export const generateForecast = async (modelType, config) => {
    const { selectedTicker, forecastHorizon, targetCol = 'Close' } = config;

    try {
        console.log(`Sending forecast request to: ${API_GATEWAY_URL}/predict/${modelType}`);
        const response = await axios.get(`${API_GATEWAY_URL}/predict/${modelType}`, {
            params: {
                ticket: selectedTicker,
                forecast_horizon: forecastHorizon,
                target_col: targetCol
            }
        });
        console.log("Forecast response:", response.data);
        if (!response.data.historical_dates || !response.data.historical_values) {
            console.warn("Backend response for /predict is missing 'historical_dates' or 'historical_values'. The graph may not display correctly.");
            // Podrías intentar devolver datos vacíos o manejar el error de otra forma
             response.data.historical_dates = response.data.historical_dates || [];
             response.data.historical_values = response.data.historical_values || [];

        }
        return response.data;
    } catch (error) {
        console.error("Error generating forecast:", error.response ? error.response.data : error.message);
         throw error.response ? new Error(JSON.stringify(error.response.data)) : error;
    }
};

import axios from 'axios';

// URL del API Gateway
const API_GATEWAY_URL = 'http://localhost:8000'; // Ajusta si tu gateway corre en otro puerto

export const trainModel = async (modelType, config) => {
    // Extraer y validar configuración
    const {
        selectedTicker,
        startDate,
        endDate,
        nLags = 10,
        targetCol = 'Close',
        trainSize = 0.8,
        training_period_preset,
        custom_start_date,
        custom_end_date,
        // LSTM specific parameters
        sequence_length,
        epochs,
        lstm_units,
        dropout_rate,
        optimize_params,
        // Random Forest specific parameters
        rf_n_estimators,
        rf_max_depth,
        rf_min_samples_split,
        rf_min_samples_leaf,
        rf_max_features,
        rf_cv_folds,
        // General parameters
        training_period,
        save_model_path
    } = config;

    // Validar que el ticker no sea nulo o indefinido
    if (!selectedTicker) {
        const errorMessage = "Ticker faltante en la configuración.";
        console.error("Error en trainModel:", errorMessage);
        throw new Error(errorMessage);
    }

    // Determinar qué fechas usar basándose en la configuración
    let finalStartDate, finalEndDate;
    let usePreset = false;

    if (training_period_preset === 'custom' && custom_start_date && custom_end_date) {
        // Usar fechas personalizadas
        finalStartDate = custom_start_date;
        finalEndDate = custom_end_date;
        usePreset = false;
        console.log("Usando fechas personalizadas:", { custom_start_date, custom_end_date });
    } else if (training_period_preset && training_period_preset !== 'custom') {
        // Usar preset de período
        finalStartDate = startDate;
        finalEndDate = endDate;
        usePreset = true;
        console.log("Usando preset de período:", training_period_preset);
    } else {
        // Fallback a las fechas por defecto
        finalStartDate = startDate;
        finalEndDate = endDate;
        usePreset = false;
        console.log("Usando fechas por defecto:", { startDate, endDate });
    }

    // Validar que las fechas finales sean válidas
    if (!finalStartDate || !finalEndDate) {
        const errorMessage = "Fechas de inicio o fin faltantes o inválidas en la configuración.";
        console.error("Error en trainModel:", errorMessage);
        throw new Error(errorMessage);
    }    // Formatear fechas a YYYY-MM-DD
    const startStr = finalStartDate.toISOString().split('T')[0];
    const endStr = finalEndDate.toISOString().split('T')[0];

    // Crear objeto payload que replica la estructura de TrainRequest
    const payload = {
        ticket: selectedTicker,
        start_date: startStr,
        end_date: endStr,
        training_period: training_period,
        n_lags: nLags,
        target_col: targetCol,
        train_size: trainSize,
        save_model_path: save_model_path,
        
        // LSTM specific parameters
        sequence_length: sequence_length,
        epochs: epochs,
        lstm_units: lstm_units,
        dropout_rate: dropout_rate,
        optimize_params: optimize_params,
        
        // Random Forest specific parameters
        rf_n_estimators: rf_n_estimators,
        rf_max_depth: rf_max_depth,
        rf_min_samples_split: rf_min_samples_split,
        rf_min_samples_leaf: rf_min_samples_leaf,
        rf_max_features: rf_max_features,
        rf_cv_folds: rf_cv_folds
    };

    // Limpieza de valores undefined para evitar enviar campos innecesarios
    Object.keys(payload).forEach(key => {
        if (payload[key] === undefined) {
            delete payload[key];
        }
    });

    // Construir la URL del endpoint
    const url = `${API_GATEWAY_URL}/train/${modelType}`;    try {
        console.log(`Enviando solicitud de entrenamiento a: ${url}`);
        console.log("Datos a enviar (payload):", payload);

        // Enviar la solicitud POST con JSON payload
        const response = await axios.post(url, payload, {
            headers: {
                'Content-Type': 'application/json'
            }
        });

        console.log("Respuesta del entrenamiento:", response.data);
        return response.data;} catch (error) {
        // Enhanced error handling to preserve response structure
        console.error("Error training model:", error.response ? error.response.data : error.message);
        
        if (error.response) {
            // Server responded with an error status
            console.error("Error Response Data:", error.response.data);
            console.error("Error Response Status:", error.response.status);
            console.error("Error Response Headers:", error.response.headers);
            
            // Create enhanced error with preserved response structure
            const errorMessage = error.response.data?.detail || error.response.data?.message || 
                               (typeof error.response.data === 'object' ? JSON.stringify(error.response.data) : String(error.response.data));
            
            const enhancedError = new Error(`Error en el entrenamiento: ${errorMessage}`);
            enhancedError.response = error.response; // Preserve the full response object
            enhancedError.status = error.response.status; // Preserve status code
            throw enhancedError;
        } else if (error.request) {
            // No response received
            console.error("Error Request:", error.request);
            const noResponseError = new Error("No se recibió respuesta del servidor. Verifica que el API Gateway esté corriendo.");
            noResponseError.isNetworkError = true;
            throw noResponseError;
        } else {
            // Request setup error
            console.error('Error Message:', error.message);
            throw new Error(`Error configurando la solicitud: ${error.message}`);
        }
    }
};


export const getTrainingStatus = async (modelType, jobId) => {
    // Validar parámetros requeridos
    if (!modelType) {
        const errorMessage = "modelType es requerido para obtener el estado del entrenamiento.";
        console.error("Error en getTrainingStatus:", errorMessage);
        throw new Error(errorMessage);
    }

    if (!jobId) {
        const errorMessage = "jobId es requerido para obtener el estado del entrenamiento.";
        console.error("Error en getTrainingStatus:", errorMessage);
        throw new Error(errorMessage);
    }

    // Construir la URL del endpoint
    const url = `${API_GATEWAY_URL}/train_status/${modelType}/${jobId}`;

    try {
        console.log(`Consultando estado del entrenamiento: ${url}`);
        
        // Enviar la solicitud GET
        const response = await axios.get(url);

        console.log("Respuesta del estado del entrenamiento:", response.data);
        return response.data;    } catch (error) {
        // Enhanced error handling to preserve response structure
        console.error("Error checking training status:", error.response ? error.response.data : error.message);
        
        if (error.response) {
            // Server responded with an error status
            console.error("Error Response Data:", error.response.data);
            console.error("Error Response Status:", error.response.status);
            console.error("Error Response Headers:", error.response.headers);
            
            // Create enhanced error with preserved response structure
            const errorMessage = error.response.data?.detail || error.response.data?.message || 
                               (typeof error.response.data === 'object' ? JSON.stringify(error.response.data) : String(error.response.data));
            
            const enhancedError = new Error(`Error consultando el estado del entrenamiento: ${errorMessage}`);
            enhancedError.response = error.response; // Preserve the full response object
            enhancedError.status = error.response.status; // Preserve status code
            throw enhancedError;
        } else if (error.request) {
            // No response received
            console.error("Error Request:", error.request);
            const noResponseError = new Error("No se recibió respuesta del servidor. Verifica que el API Gateway esté corriendo.");
            noResponseError.isNetworkError = true;
            throw noResponseError;
        } else {
            // Request setup error
            console.error('Error Message:', error.message);
            throw new Error(`Error configurando la solicitud: ${error.message}`);
        }
    }
};


export const generateForecast = async (modelType, config) => {
    const { selectedTicker, forecastHorizon, targetCol = 'Close' } = config;

    // Validate required parameters
    if (!modelType) {
        const errorMessage = "modelType es requerido para generar pronóstico.";
        console.error("Error en generateForecast:", errorMessage);
        throw new Error(errorMessage);
    }

    if (!selectedTicker) {
        const errorMessage = "selectedTicker es requerido para generar pronóstico.";
        console.error("Error en generateForecast:", errorMessage);
        throw new Error(errorMessage);
    }

    if (!forecastHorizon || forecastHorizon <= 0) {
        const errorMessage = "forecastHorizon debe ser un número positivo.";
        console.error("Error en generateForecast:", errorMessage);
        throw new Error(errorMessage);
    }

    // Construct the URL
    const url = `${API_GATEWAY_URL}/predict/${modelType}`;

    try {
        console.log(`Sending forecast request to: ${url}`);
        console.log("Forecast parameters:", { 
            ticket: selectedTicker, 
            forecast_horizon: forecastHorizon, 
            target_col: targetCol 
        });

        const response = await axios.get(url, {
            params: {
                ticket: selectedTicker,
                forecast_horizon: forecastHorizon,
                target_col: targetCol
            }
        });

        console.log("Forecast response:", response.data);
        
        // Validate response structure and provide fallbacks if needed
        if (!response.data.historical_dates || !response.data.historical_values) {
            console.warn("Backend response for /predict is missing 'historical_dates' or 'historical_values'. The graph may not display correctly.");
            response.data.historical_dates = response.data.historical_dates || [];
            response.data.historical_values = response.data.historical_values || [];
        }
        
        return response.data;
    } catch (error) {
        // Enhanced error handling to preserve response structure
        console.error("Error generating forecast:", error.response ? error.response.data : error.message);
        
        if (error.response) {
            // Server responded with an error status
            console.error("Error Response Data:", error.response.data);
            console.error("Error Response Status:", error.response.status);
            console.error("Error Response Headers:", error.response.headers);
            
            // Create enhanced error with preserved response structure
            const errorMessage = error.response.data?.detail || error.response.data?.message || 
                               (typeof error.response.data === 'object' ? JSON.stringify(error.response.data) : String(error.response.data));
            
            const enhancedError = new Error(`Error generando pronóstico: ${errorMessage}`);
            enhancedError.response = error.response; // Preserve the full response object
            enhancedError.status = error.response.status; // Preserve status code
            throw enhancedError;
        } else if (error.request) {
            // No response received
            console.error("Error Request:", error.request);
            const noResponseError = new Error("No se recibió respuesta del servidor. Verifica que el API Gateway esté corriendo.");
            noResponseError.isNetworkError = true;
            throw noResponseError;
        } else {
            // Request setup error
            console.error('Error Message:', error.message);
            throw new Error(`Error configurando la solicitud: ${error.message}`);
        }
    }
};

export const getAvailableModels = async (modelType) => {
    // Validar parámetros requeridos
    if (!modelType) {
        throw new Error("El tipo de modelo (modelType) es requerido.");
    }

    // Construir la URL del endpoint
    const url = `${API_GATEWAY_URL}/models/${modelType}`;

    try {
        console.log(`Solicitando modelos disponibles para: ${modelType}`);
        console.log(`URL: ${url}`);

        // Enviar la solicitud GET
        const response = await axios.get(url, {
            headers: {
                'Content-Type': 'application/json',
            },
            timeout: 30000, // 30 segundos de timeout
        });

        console.log(`Modelos disponibles recibidos para ${modelType}:`, response.data);
        return response.data;

    } catch (error) {
        console.error(`Error obteniendo modelos disponibles para ${modelType}:`, error);

        if (error.response) {
            // Server responded with error status
            const errorMessage = error.response.data?.detail || 
                                error.response.data?.message || 
                                `Error ${error.response.status}: ${error.response.statusText}`;
            console.error("Error Response Data:", error.response.data);
            console.error("Error Response Status:", error.response.status);
            throw new Error(`Error del servidor: ${errorMessage}`);
        } else if (error.request) {
            // No response received
            console.error("Error Request:", error.request);
            const noResponseError = new Error("No se recibió respuesta del servidor. Verifica que el API Gateway esté corriendo.");
            noResponseError.isNetworkError = true;
            throw noResponseError;
        } else {
            // Request setup error
            console.error('Error Message:', error.message);
            throw new Error(`Error configurando la solicitud: ${error.message}`);
        }
    }
};


// ============== SHAP EXPLAINER API ==============

/**
 * Get SHAP explanations for model predictions
 * @param {string} ticker - Stock ticker symbol
 * @param {string} modelType - Model type: 'xgboost' or 'rf'
 * @param {number} topFeatures - Number of top features to return
 */
export const getShapExplanation = async (ticker, modelType = 'xgboost', topFeatures = 10) => {
    const url = `${API_GATEWAY_URL}/explain`;
    
    try {
        console.log(`Requesting SHAP explanation for ${ticker} using ${modelType}`);
        
        const response = await axios.post(url, {
            ticker: ticker,
            model_type: modelType,
            top_features: topFeatures
        }, {
            headers: { 'Content-Type': 'application/json' },
            timeout: 60000
        });
        
        console.log('SHAP explanation received:', response.data);
        return response.data;
        
    } catch (error) {
        console.error('Error getting SHAP explanation:', error);
        throw handleApiError(error, 'SHAP explanation');
    }
};

/**
 * Get global feature importance using SHAP values
 * @param {string} modelType - Model type: 'xgboost' or 'rf'
 * @param {string} ticker - Stock ticker symbol
 * @param {number} maxSamples - Maximum samples for calculation
 */
export const getGlobalImportance = async (modelType, ticker = 'NU', maxSamples = 500) => {
    const url = `${API_GATEWAY_URL}/explain/importance/${modelType}`;
    
    try {
        console.log(`Requesting global importance for ${modelType}`);
        
        const response = await axios.get(url, {
            params: { ticker, max_samples: maxSamples },
            timeout: 90000
        });
        
        console.log('Global importance received:', response.data);
        return response.data;
        
    } catch (error) {
        console.error('Error getting global importance:', error);
        throw handleApiError(error, 'global importance');
    }
};

/**
 * Get SHAP summary plot as base64 image
 * @param {string} modelType - Model type: 'xgboost' or 'rf'
 * @param {string} ticker - Stock ticker symbol
 * @param {string} plotType - 'bar' or 'dot'
 * @param {number} maxFeatures - Maximum features to display
 */
export const getShapPlot = async (modelType, ticker = 'NU', plotType = 'bar', maxFeatures = 15) => {
    const url = `${API_GATEWAY_URL}/explain/plot/${modelType}`;
    
    try {
        console.log(`Requesting SHAP plot for ${modelType}`);
        
        const response = await axios.get(url, {
            params: { ticker, plot_type: plotType, max_features: maxFeatures },
            timeout: 90000
        });
        
        console.log('SHAP plot received');
        return response.data;
        
    } catch (error) {
        console.error('Error getting SHAP plot:', error);
        throw handleApiError(error, 'SHAP plot');
    }
};

/**
 * Get SHAP waterfall plot for a single prediction
 * @param {string} modelType - Model type: 'xgboost' or 'rf'
 * @param {string} ticker - Stock ticker symbol
 * @param {number} predictionIndex - Index of prediction to explain
 * @param {number} maxFeatures - Maximum features to display
 */
export const getWaterfallPlot = async (modelType, ticker = 'NU', predictionIndex = 0, maxFeatures = 10) => {
    const url = `${API_GATEWAY_URL}/explain/waterfall/${modelType}`;
    
    try {
        console.log(`Requesting waterfall plot for ${modelType}, prediction ${predictionIndex}`);
        
        const response = await axios.get(url, {
            params: { ticker, prediction_index: predictionIndex, max_features: maxFeatures },
            timeout: 90000
        });
        
        console.log('Waterfall plot received');
        return response.data;
        
    } catch (error) {
        console.error('Error getting waterfall plot:', error);
        throw handleApiError(error, 'waterfall plot');
    }
};


// ============== ENSEMBLE MODEL API ==============

/**
 * Get ensemble prediction combining multiple models
 * @param {Object} config - Prediction configuration
 * @param {string} config.ticker - Stock ticker symbol
 * @param {number} config.forecastHorizon - Number of days to forecast
 * @param {string} config.targetCol - Target column (default: 'Close')
 * @param {Array<string>} config.models - Models to use (null = all)
 * @param {string} config.ensembleMethod - Ensemble method
 */
export const getEnsemblePrediction = async (config) => {
    const {
        ticker = 'NU',
        forecastHorizon = 10,
        targetCol = 'Close',
        models = null,
        ensembleMethod = 'weighted_average'
    } = config;
    
    const url = `${API_GATEWAY_URL}/ensemble/predict`;
    
    try {
        console.log(`Requesting ensemble prediction for ${ticker}`);
        
        const response = await axios.post(url, {
            ticker: ticker,
            forecast_horizon: forecastHorizon,
            target_col: targetCol,
            models: models,
            ensemble_method: ensembleMethod
        }, {
            headers: { 'Content-Type': 'application/json' },
            timeout: 120000 // 2 minutes timeout for ensemble
        });
        
        console.log('Ensemble prediction received:', response.data);
        return response.data;
        
    } catch (error) {
        console.error('Error getting ensemble prediction:', error);
        throw handleApiError(error, 'ensemble prediction');
    }
};

/**
 * Compare predictions from all models
 * @param {string} ticker - Stock ticker symbol
 * @param {number} forecastHorizon - Number of days to forecast
 * @param {string} targetCol - Target column
 */
export const compareModelPredictions = async (ticker = 'NU', forecastHorizon = 10, targetCol = 'Close') => {
    const url = `${API_GATEWAY_URL}/ensemble/compare`;
    
    try {
        console.log(`Comparing model predictions for ${ticker}`);
        
        const response = await axios.get(url, {
            params: { ticker, forecast_horizon: forecastHorizon, target_col: targetCol },
            timeout: 120000
        });
        
        console.log('Model comparison received:', response.data);
        return response.data;
        
    } catch (error) {
        console.error('Error comparing models:', error);
        throw handleApiError(error, 'model comparison');
    }
};

/**
 * Get available ensemble models and their status
 */
export const getEnsembleModels = async () => {
    const url = `${API_GATEWAY_URL}/ensemble/models`;
    
    try {
        const response = await axios.get(url, { timeout: 30000 });
        console.log('Ensemble models received:', response.data);
        return response.data;
        
    } catch (error) {
        console.error('Error getting ensemble models:', error);
        throw handleApiError(error, 'ensemble models');
    }
};


// ============== HELPER FUNCTIONS ==============

/**
 * Handle API errors consistently
 */
const handleApiError = (error, operationName) => {
    if (error.response) {
        const errorMessage = error.response.data?.detail || 
                            error.response.data?.message || 
                            `Error ${error.response.status}: ${error.response.statusText}`;
        const enhancedError = new Error(`Error en ${operationName}: ${errorMessage}`);
        enhancedError.status = error.response.status;
        enhancedError.data = error.response.data;
        return enhancedError;
    } else if (error.request) {
        const noResponseError = new Error(`No se recibió respuesta del servidor para ${operationName}. Verifica que el servicio esté corriendo.`);
        noResponseError.isNetworkError = true;
        return noResponseError;
    } else {
        return new Error(`Error en ${operationName}: ${error.message}`);
    }
};
// Test script for extractTickerFromModelName function
const extractTickerFromModelName = (modelName) => {
    if (!modelName || typeof modelName !== 'string') return null;
    
    const patterns = [
        /^lstm_model_([A-Z]{1,5})$/i,  // LSTM directory format: lstm_model_TICKER
        /_([A-Z]{1,5})\.(?:keras|joblib|pkl|h5)$/i,  // Standard format: model_TICKER.extension
        /_([A-Z]{1,5})_\d+\.(?:keras|joblib|pkl|h5)$/i,  // With date: model_TICKER_date.extension
        /([A-Z]{1,5})_model/i,  // Prefix format: TICKER_model
        /_([A-Z]{1,5})$/i,  // Simple underscore format: whatever_TICKER
        /([A-Z]{2,5})(?:_|\.|$)/i  // Any uppercase 2-5 letters followed by separator or end
    ];
    
    for (const pattern of patterns) {
        const match = modelName.match(pattern);
        if (match && match[1]) {
            return match[1].toUpperCase();
        }
    }
    
    return null;
};

// Test cases
const testCases = [
    'lstm_model_AAPL',
    'lstm_model_TSLA', 
    'lstm_model_MSFT',
    'rf_model_AAPL.joblib',
    'xgb_model_TSLA.pkl',
    'model_MSFT_20231201.h5',
    'lstm_model_generic',
    'lstm_model_NU',
    'LSTM',  // This should NOT extract "LSTM" as ticker
    'lstm_model',  // This should return null
    'lstm_model_GOOGL'
];

console.log('=== Pruebas de extractTickerFromModelName ===');
testCases.forEach(testCase => {
    const result = extractTickerFromModelName(testCase);
    console.log(`${testCase} -> ${result}`);
});

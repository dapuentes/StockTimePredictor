import React from 'react';
import { Descriptions, Tooltip, Tag } from 'antd'; 
import { QuestionCircleOutlined } from '@ant-design/icons';

const hyperparameterInterpretations = {
    max_depth: "Profundidad máxima de cada árbol en el bosque. Controla qué tan profundos serán los árboles: valores más altos permiten capturar relaciones complejas, pero aumentan el riesgo de sobreajuste y el tiempo de entrenamiento.",
  
    max_features: "Número máximo de características a considerar en cada división de nodo. Un valor menor reduce la correlación entre árboles y mejora la diversidad del bosque, mientras que uno mayor puede acelerar la convergencia pero favorecer el sobreajuste.",

    min_samples_leaf: "Cantidad mínima de muestras que debe tener un nodo hoja. Aumentar este parámetro hace los árboles más conservadores, evitando divisiones basadas en muy pocos datos, lo que mejora la generalización.",

    min_samples_split: "Número mínimo de muestras requeridas para dividir un nodo interno en dos. Valores más altos evitan divisiones con datos escasos, reduciendo la varianza del modelo pero pudiendo pasar por alto patrones finos.",

    n_estimators: "Cantidad de árboles que se construyen en el bosque aleatorio. Más árboles suelen mejorar la estabilidad y precisión de las predicciones, a costa de un mayor tiempo de entrenamiento y de inferencia.",

    features_index: "Índices de las características seleccionadas por el modelo o por el mecanismo de selección de variables. Muestra qué columnas del conjunto de datos aportan más valor predictivo.",

    lstm_units_1: "Número de neuronas en la primera capa LSTM. Más unidades mejoran la capacidad para aprender patrones de largo plazo, pero aumentan el coste computacional y el riesgo de sobreajuste.",

    lstm_units_2: "Número de neuronas en la segunda capa LSTM. Esta capa refina lo aprendido por la primera, capturando interacciones más complejas entre pasos de la serie temporal.",

    dropout_rate: "Proporción de neuronas que se ‘apagan’ aleatoriamente durante el entrenamiento. Un valor moderado (p. ej. 0.2) ayuda a evitar dependencias excesivas entre neuronas y mejora la robustez del modelo.",

    learning_rate: "Tamaño del paso de actualización de los pesos del modelo en cada iteración. Tasas más altas aceleran el aprendizaje pero pueden causar inestabilidad; tasas más bajas ofrecen mayor precisión, pero requieren más tiempo de entrenamiento."
};

/**
 * Component to display details about the latest model run.
 *
 * @param {Object} props - The component props.
 * @param {Object} props.latestRun - The latest run data containing model details.
 * @param {string} [props.latestRun.modelType] - The type of the model.
 * @param {string} [props.latestRun.ticker] - The ticker associated with the model.
 * @param {string} [props.latestRun.modelPath] - The specific path or identifier of the model.
 * @param {Object} [props.latestRun.bestParams] - The optimized hyperparameters for the model.
 * @param {Array<string>} [props.latestRun.featureNames] - The names of features used in the model.
 * @param {string} [props.latestRun.dateRange] - The date range associated with the model run.
 *
 * @returns {JSX.Element} A React component displaying model details or a message if no data is available.
 */
function ModelDetailsDisplay({ latestRun }) {

    // Si no hay datos de una ejecución, muestra un mensaje
    if (!latestRun) {
        return <p>No hay detalles de modelo disponibles. Entrena o pronostica primero.</p>;
    }

    // Extrae los datos necesarios de latestRun
    const { modelType, ticker, modelPath, bestParams = {}, featureNames = [] } = latestRun;

    return (
        <div>
            <p>Información sobre el modelo:</p>
            <ul>
                <li><strong>Tipo:</strong> {modelType || 'N/A'}</li>
                <li><strong>Ticker Entrenado/Usado:</strong> {ticker || 'N/A'}</li>
                <li><strong>Modelo Específico:</strong> {modelPath || 'N/A'}</li>
                <li><strong>Rango de Fechas:</strong> {latestRun.dateRange || 'N/A'}</li>
            </ul>

            {Object.keys(bestParams).length > 0 && (
                <>
                    <h4 style={{ marginTop: '16px' }}>Hiperparámetros Optimizados:</h4>
                    <Descriptions bordered size="small" column={1}>
                        {Object.entries(bestParams).map(([key, value]) => {
                            const cleanKey = key.replace('rf__', '').replace('selector__', '');
                            let displayValue = String(value);
                            const interpretation = hyperparameterInterpretations[cleanKey] || 'Parámetro específico del modelo.';

                            if (key === 'selector__features_index' && Array.isArray(value) && featureNames.length > 0) {
                                try {
                                    const selectedNames = value.map(index => featureNames[index] || `Índice inválido: ${index}`);
                                    displayValue = (
                                        <ul style={{ margin: 0, paddingLeft: '20px', listStyle: 'disc' }}>
                                            {selectedNames.map((name, i) => <li key={`<span class="math-inline">\{name\}\-</span>{i}`}>{name}</li>)}
                                        </ul>
                                    );
                                } catch (e) { /* ... manejo error ... */ }
                            } else if (typeof value === 'string' && (value.startsWith('[') || value.startsWith('{'))) {
                                displayValue = value;
                            }

                            return (
                                <Descriptions.Item
                                    key={key}
                                    label={
                                        <span>
                                            {cleanKey}{' '}
                                            <Tooltip 
                                            title={interpretation}
                                            overlayStyle={{ maxWidth: '450px' }}
                                            >
                                                <QuestionCircleOutlined style={{ color: 'rgba(0,0,0,.45)', cursor: 'help' }} />
                                            </Tooltip>
                                        </span>
                                    }
                                >
                                    {displayValue}
                                </Descriptions.Item>
                            );
                        })}
                    </Descriptions>
                </>
            )}
            {Object.keys(bestParams).length === 0 && (
                 <p style={{ marginTop: '16px' }}><i>No hay hiperparámetros optimizados disponibles para esta ejecución.</i></p>
            )}
        </div>
    );
}

export default ModelDetailsDisplay;
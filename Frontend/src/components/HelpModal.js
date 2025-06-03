import { Modal, Typography, Collapse, Divider } from 'antd';
const { Title, Paragraph, Text } = Typography;
const { Panel } = Collapse;

function HelpModal({ visible, onClose }) {
    return (
        <Modal 
            title="Ayuda y Tutorial" 
            open={visible} 
            onCancel={onClose} 
            footer={null} 
            width={800}
        >
            <Typography>
                <Title level={4}>Guía de Uso</Title>
                <Paragraph>
                    Bienvenido/a al sistema de pronóstico financiero. Este sistema está diseñado para ser <Text strong>intuitivo y eficiente</Text>: 
                    simplemente selecciona una acción y genera un pronóstico. Si no existe un modelo entrenado, el sistema te ofrecerá crearlo automáticamente.
                </Paragraph>
                
                <Collapse accordion>
                    <Panel header="🚀 Flujo Principal: Generar Pronóstico" key="1">
                        <Paragraph><Text strong>Paso 1:</Text> Selecciona el <Text code>Ticker</Text> (símbolo de la acción, ej. AAPL, NU, TSLA)</Paragraph>
                        <Paragraph><Text strong>Paso 2:</Text> Selecciona el <Text code>Tipo de Modelo</Text> (RF, LSTM, XGBoost, Prophet)</Paragraph>
                        <Paragraph><Text strong>Paso 3:</Text> Configura el <Text code>Horizonte de Pronóstico</Text> (días a predecir)</Paragraph>
                        <Paragraph><Text strong>Paso 4:</Text> Haz clic en <Text strong style={{color: '#1890ff'}}>📊 Generar Pronóstico</Text></Paragraph>
                        <Paragraph><Text type="success">✅ Si existe un modelo entrenado, obtendrás el pronóstico inmediatamente</Text></Paragraph>
                        <Paragraph><Text type="warning">⚠️ Si no existe un modelo, el sistema te preguntará si deseas entrenarlo automáticamente</Text></Paragraph>
                        <Paragraph><Text type="secondary">ℹ️ El sistema pronostica automáticamente el precio de cierre (Close)</Text></Paragraph>
                    </Panel>
                    
                    <Panel header="⚙️ Gestión Avanzada de Modelos" key="2">
                        <Paragraph>Para usuarios avanzados que deseen controlar el entrenamiento manualmente:</Paragraph>
                        <Paragraph><Text strong>Período de Entrenamiento:</Text> Selecciona el rango de datos (1 año, 3 años, 5 años, o personalizado)</Paragraph>
                        <Paragraph><Text strong>Número de Lags:</Text> Características temporales del modelo (valores anteriores a considerar)</Paragraph>
                        <Paragraph><Text strong>Entrenamiento Manual:</Text> Usa el botón <Text code>🎯 Entrenar/Actualizar Modelo</Text> para crear o actualizar modelos con configuraciones específicas</Paragraph>
                        <Paragraph><Text type="info">💡 El entrenamiento es una tarea asíncrona - puedes seguir el progreso en tiempo real</Text></Paragraph>
                    </Panel>
                    
                    <Panel header="📊 Interpretación de Métricas" key="3">
                        <Paragraph><Text strong>MAPE:</Text> Error Porcentual Absoluto Medio. Un valor más bajo es mejor. Indica el error promedio como porcentaje.</Paragraph>
                        <Paragraph><Text strong>RMSE:</Text> Raíz del Error Cuadrático Medio. Un valor más bajo es mejor. Indica la magnitud del error promedio.</Paragraph>
                        <Paragraph><Text strong>MAE:</Text> Error Absoluto Medio. Un valor más bajo es mejor. Indica el error promedio absoluto.</Paragraph>
                        <Paragraph><Text strong>MSE:</Text> Error Cuadrático Medio. Medida de la varianza del error.</Paragraph>
                    </Panel>
                    
                    <Panel header="🔄 Tipos de Modelos Disponibles" key="4">
                        <Paragraph><Text strong>Random Forest (RF):</Text> Modelo robusto y rápido, ideal para datos con patrones complejos</Paragraph>
                        <Paragraph><Text strong>LSTM:</Text> Red neuronal especializada en secuencias temporales, potente para patrones a largo plazo</Paragraph>
                        <Paragraph><Text strong>XGBoost:</Text> Algoritmo de gradient boosting, excelente balance entre precisión y velocidad</Paragraph>
                        <Paragraph><Text strong>Prophet:</Text> Modelo de Facebook optimizado para series temporales con estacionalidad</Paragraph>
                    </Panel>
                </Collapse>
                
                <Divider />
                <Paragraph>
                    <Text strong>💡 Consejos importantes:</Text>
                </Paragraph>
                <Paragraph>
                    • Los pronósticos son estimaciones basadas en patrones históricos, no garantías de resultados futuros<br/>
                    • Se recomienda un mínimo de 760 días (~25 meses) de datos históricos para obtener mejores resultados<br/>
                    • Puedes experimentar con diferentes modelos y comparar sus métricas para encontrar el más adecuado<br/>
                    • El sistema maneja automáticamente la limpieza y preparación de datos
                </Paragraph>
            </Typography>
        </Modal>
    );
}

export default HelpModal;
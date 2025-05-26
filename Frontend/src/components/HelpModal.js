import { Modal, Typography, Collapse, Divider } from 'antd';
const { Title, Paragraph, Text } = Typography;
const { Panel } = Collapse;

function HelpModal({ visible, onClose }) {
    return (
      <Modal title="Ayuda y Tutorial" open={visible} onCancel={onClose} footer={null} width={800}>
        <Typography>
          <Title level={4}>Guía de Uso</Title>
          <Paragraph>Bienvenido/a al sistema de pronóstico financiero...</Paragraph>
          <Collapse accordion>
            <Panel header="Configuración de Parámetros" key="1">
              <Paragraph><Text strong>Ticker:</Text> Símbolo de la acción (ej. AAPL, NU).</Paragraph>
                <Paragraph><Text strong>Rango de Fechas:</Text> Selecciona el rango de fechas para el entrenamiento del modelo.</Paragraph>
                <Paragraph><Text strong>Modelo:</Text> Selecciona el tipo de modelo a utilizar (ej. RF, LSTM).</Paragraph>
                <Paragraph><Text strong>Número de Lags:</Text> Número de lags a considerar en el modelo.</Paragraph>
                <Paragraph><Text strong>Horizonte de Pronóstico:</Text> Número de días a pronosticar.</Paragraph>
            </Panel>
            <Panel header="Interpretación de Métricas" key="2">
              <Paragraph><Text strong>MAPE:</Text> Error Porcentual Absoluto Medio. Un valor más bajo es mejor. Indica el error promedio como porcentaje.</Paragraph>
                <Paragraph><Text strong>RMSE:</Text> Raíz del Error Cuadrático Medio. Un valor más bajo es mejor. Indica la magnitud del error promedio.</Paragraph>
                <Paragraph><Text strong>MAE:</Text> Error Absoluto Medio. Un valor más bajo es mejor. Indica el error promedio absoluto.</Paragraph>
                <Paragraph><Text strong>MSE:</Text> Error Cuadrático Medio</Paragraph>
            </Panel>
          </Collapse>
          <Divider />
          <Paragraph>Recuerda que los pronósticos son estimaciones y no garantías.</Paragraph>
        </Typography>
      </Modal>
    );
  }
  export default HelpModal;
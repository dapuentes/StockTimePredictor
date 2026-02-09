/**
 * WelcomeCard — Empty-state contextual onboarding
 * Shown inside the Results panel when no data exists yet.
 * Auto-hides once the user trains a model or generates a forecast.
 */
import React from 'react';
import { Typography, Steps } from 'antd';
import {
  StockOutlined,
  ExperimentOutlined,
  LineChartOutlined,
  RocketOutlined,
} from '@ant-design/icons';

const { Title, Paragraph, Text } = Typography;

function WelcomeCard({ selectedTicker, selectedModelType }) {
  return (
    <div className="welcome-card">
      <div className="welcome-card__icon">
        <RocketOutlined />
      </div>

      <Title level={3} className="welcome-card__title">
        Bienvenido a StockTime Predictor
      </Title>

      <Paragraph className="welcome-card__description">
        Plataforma de predicción de series financieras con modelos de Machine Learning.
        <br />
        Entrena modelos, genera pronósticos y compara resultados en un solo lugar.
      </Paragraph>

      <div className="welcome-card__steps">
        <Steps
          direction="vertical"
          size="small"
          current={selectedTicker ? (selectedModelType ? 2 : 1) : 0}
          items={[
            {
              title: 'Selecciona un ticker',
              description: selectedTicker
                ? <Text type="success">{selectedTicker} seleccionado</Text>
                : 'Elige la acción que deseas analizar en el panel de configuración',
              icon: <StockOutlined />,
            },
            {
              title: 'Elige un modelo',
              description: selectedModelType
                ? <Text type="success">{selectedModelType.toUpperCase()} seleccionado</Text>
                : 'Random Forest, LSTM, XGBoost o Prophet',
              icon: <ExperimentOutlined />,
            },
            {
              title: 'Entrena o pronostica',
              description: 'Usa los botones de la barra superior para lanzar tu primer análisis',
              icon: <LineChartOutlined />,
            },
          ]}
        />
      </div>

      <div className="welcome-card__hint">
        <Text type="secondary">
          Consejo: Si ya existe un modelo entrenado, puedes generar un pronóstico directamente sin entrenar de nuevo.
        </Text>
      </div>
    </div>
  );
}

export default WelcomeCard;

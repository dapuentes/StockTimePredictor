/**
 * QuickActions Component — Barra de acciones rápidas
 * Usa clases CSS de globals.css (quick-actions*) — sin inline styles.
 */
import React from 'react';
import { Button, Dropdown, Tooltip } from 'antd';
import {
  ThunderboltOutlined,
  LineChartOutlined,
  DownloadOutlined,
  ReloadOutlined,
  CaretDownOutlined,
  RocketOutlined
} from '@ant-design/icons';

function QuickActions({
  onForecast,
  onTrain,
  onExport,
  onRefreshModels,
  isTraining,
  isForecastPending,
  selectedTicker,
  selectedModelType,
  hasForecastData
}) {
  const trainMenuItems = [
    {
      key: 'rf',
      label: 'Random Forest',
      icon: <ThunderboltOutlined />,
      onClick: () => onTrain('rf')
    },
    {
      key: 'lstm',
      label: 'LSTM Neural Network',
      icon: <ThunderboltOutlined />,
      onClick: () => onTrain('lstm')
    },
    {
      key: 'xgboost',
      label: 'XGBoost',
      icon: <ThunderboltOutlined />,
      onClick: () => onTrain('xgboost')
    },
    {
      key: 'prophet',
      label: 'Prophet',
      icon: <ThunderboltOutlined />,
      onClick: () => onTrain('prophet')
    }
  ];

  return (
    <div className="quick-actions">
      <Tooltip title={`Pronosticar con ${selectedModelType?.toUpperCase()} para ${selectedTicker}`}>
        <Button
          type="primary"
          icon={<LineChartOutlined />}
          onClick={onForecast}
          loading={isForecastPending}
          disabled={!selectedTicker || isForecastPending}
        >
          {isForecastPending ? 'Generando...' : 'Pronosticar'}
        </Button>
      </Tooltip>

      <Dropdown
        menu={{ items: trainMenuItems }}
        trigger={['click']}
        disabled={!selectedTicker || isTraining}
      >
        <Button icon={<RocketOutlined />} loading={isTraining}>
          Entrenar Modelo <CaretDownOutlined />
        </Button>
      </Dropdown>

      <div className="quick-actions__divider" />

      <Tooltip title="Exportar pronóstico a CSV">
        <Button
          icon={<DownloadOutlined />}
          onClick={onExport}
          disabled={!hasForecastData}
        >
          Exportar
        </Button>
      </Tooltip>

      <Tooltip title="Actualizar lista de modelos disponibles">
        <Button icon={<ReloadOutlined />} onClick={onRefreshModels} />
      </Tooltip>
    </div>
  );
}

export default QuickActions;

/**
 * StatsOverview Component — Tarjetas de resumen
 * Usa clases CSS de globals.css (stat-card*) — cero inline styles.
 */
import React from 'react';
import { Row, Col, Tag, Tooltip } from 'antd';
import {
  StockOutlined,
  RocketOutlined,
  LineChartOutlined,
  ThunderboltOutlined
} from '@ant-design/icons';

function StatsOverview({
  selectedTicker,
  selectedModelType,
  activeJobsCount,
  forecastHorizon
}) {
  const stats = [
    {
      key: 'ticker',
      title: 'Ticker Activo',
      value: selectedTicker || 'N/A',
      icon: <StockOutlined />,
      iconClass: 'stat-card__icon--primary',
      description: 'Acción seleccionada'
    },
    {
      key: 'model',
      title: 'Modelo',
      value: selectedModelType?.toUpperCase() || 'N/A',
      icon: <ThunderboltOutlined />,
      iconClass: 'stat-card__icon--primary',
      description: 'Tipo de modelo ML'
    },
    {
      key: 'active',
      title: 'Entrenamientos',
      value: activeJobsCount || 0,
      icon: <RocketOutlined spin={activeJobsCount > 0} />,
      iconClass: activeJobsCount > 0 ? 'stat-card__icon--success' : 'stat-card__icon--primary',
      valueClass: activeJobsCount > 0 ? 'stat-card__value--accent' : '',
      description: 'En progreso',
      suffix: activeJobsCount > 0
        ? <Tag color="processing" style={{ marginLeft: 6 }}>En curso</Tag>
        : null
    },
    {
      key: 'forecast',
      title: 'Horizonte',
      value: forecastHorizon || 10,
      icon: <LineChartOutlined />,
      iconClass: 'stat-card__icon--accent',
      description: 'Días de pronóstico',
      suffix: <span className="stat-card__label" style={{ fontSize: '0.85rem', marginLeft: 4 }}>días</span>
    }
  ];

  return (
    <div className="stats-overview animate-fadeIn">
      <Row gutter={[16, 16]}>
        {stats.map(stat => (
          <Col xs={12} sm={12} md={6} key={stat.key}>
            <Tooltip title={stat.description}>
              <div className="stat-card">
                <div>
                  <div className="stat-card__label">{stat.title}</div>
                  <div className={`stat-card__value ${stat.valueClass || ''}`}>
                    {stat.value}
                    {stat.suffix}
                  </div>
                </div>
                <div className={`stat-card__icon ${stat.iconClass}`}>
                  {stat.icon}
                </div>
              </div>
            </Tooltip>
          </Col>
        ))}
      </Row>
    </div>
  );
}

export default StatsOverview;

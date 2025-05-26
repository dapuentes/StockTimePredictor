import React, { useState } from 'react';
import { Line, Bar } from 'react-chartjs-2';
import { Card, Button, Modal, Row, Col, Empty } from 'antd';
import { QuestionCircleOutlined } from '@ant-design/icons';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement, 
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import annotationPlugin from 'chartjs-plugin-annotation';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  annotationPlugin
);

/**
 * Componente para renderizar un gráfico de autocorrelación (ACF o PACF).
 * @param {object} props - Propiedades del componente.
 * @param {string} props.title - Título del gráfico.
 * @param {object} props.correlationData - Datos de correlación que incluyen valores y bandas de confianza.
 * @returns {JSX.Element}
 */
const CorrelationChart = ({ title, correlationData }) => {
  // No renderizar si no hay datos
  if (!correlationData || !correlationData.values) {
    return <Card title={title}><Empty description="Datos no disponibles" /></Card>;
  }
  
  const isACF = title.includes('ACF');
  const labels = correlationData.values.map((_, i) => i).slice(isACF ? 1 : 0);
  const values = correlationData.values.slice(isACF ? 1 : 0);
  
  
  let finalConfintUpper = 0.2; // Fallback por defecto
  let finalConfintLower = -0.2; // Fallback por defecto

  if (correlationData.confint_upper.length > 1 && correlationData.confint_lower.length > 1) {
    finalConfintUpper = correlationData.confint_upper[1]; 
    finalConfintLower = correlationData.confint_lower[1];
  }

  const data = {
    labels: labels,
    datasets: [{
      label: title,
      data: values,
      backgroundColor: 'rgba(54, 162, 235, 0.6)',
      borderColor: 'rgba(54, 162, 235, 1)',
      borderWidth: 1,
      barPercentage: .8, 
      categoryPercentage: 0.9,
    }]
  };

  const options = {
    plugins: {
      legend: { display: false },
      title: { display: true, text: title },
      // Configuración de las bandas de confianza con el plugin de anotaciones
      annotation: {
        annotations: {
          upperBound: {
            type: 'line',
            yMin: finalConfintUpper,
            yMax: finalConfintUpper,
            borderColor: 'rgba(255, 99, 132, 0.8)',
            borderWidth: 1.5,
            borderDash: [6, 6],
          },
          lowerBound: {
            type: 'line',
            yMin: finalConfintLower,
            yMax: finalConfintLower,
            borderColor: 'rgba(255, 99, 132, 0.8)',
            borderWidth: 1.5,
            borderDash: [6, 6],
          }
        }
      }
    },
    scales: {
      x: { 
        title: { display: true, text: 'Retardo (Lag)' },
        grid: { display: false } 
      },
      y: { 
        title: { display: true, text: 'Correlación' },
        min: -.4,
        max: .4      },
    },
    maintainAspectRatio: false
  };

  return <div style={{ height: '300px' }}><Bar data={data} options={options} /></div>;
};

/**
 * Componente principal que muestra el panel de diagnóstico de residuales.
 * @param {object} props - Propiedades del componente.
 * @param {object} props.data - Objeto que contiene todos los datos de residuales.
 * @returns {JSX.Element}
 */
export default function ResidualsDisplay({ data }) {
  const [showHelp, setShowHelp] = useState(false);
  const { dates, values, acf, pacf } = data;

  // Si no hay residuales, mostrar un mensaje
  if (!values || values.length === 0) {
    return <Empty description="No hay datos de residuales. Entrena un modelo para ver los diagnósticos." />;
  }

  const timeSeriesData = {
    labels: dates,
    datasets: [{
      label: 'Residuos (y - ŷ)',
      data: values,
      borderColor: '#ff4d4f',
      backgroundColor: 'rgba(255, 77, 79, 0.2)',
      fill: true,
      tension: 0.1,
      pointRadius: 1
    }]
  };

  const timeSeriesOptions = {
    plugins: {
      legend: { display: false },
      title: { display: true, text: 'Serie Temporal de Residuales' }
    },
    scales: {
      x: { ticks: { maxRotation: 20, autoSkip: true, maxTicksLimit: 20 } },
      y: { title: { display: true, text: 'Error' } }
    },
    maintainAspectRatio: false
  };

  return (
    <Card style={{ position: 'relative' }}>
      <Button
        type="text"
        icon={<QuestionCircleOutlined />}
        onClick={() => setShowHelp(true)}
        style={{ position: 'absolute', top: 8, right: 8, zIndex: 10 }}
        aria-label="Ayuda sobre residuales"
      />

      <Row gutter={[16, 24]}>
        {/* Gráfico de la Serie Temporal de Residuales */}
        <Col span={24}>
            <div style={{ height: '250px' }}>
                <Line data={timeSeriesData} options={timeSeriesOptions} />
            </div>
        </Col>

        {/* Gráfico ACF */}
        <Col xs={24} md={12}>
            <CorrelationChart title="Función de Autocorrelación (ACF) de Residuales" correlationData={acf} />
        </Col>

        {/* Gráfico PACF */}
        <Col xs={24} md={12}>
             <CorrelationChart title="Función de Autocorrelación Parcial (PACF) de Residuales" correlationData={pacf} />
        </Col>
      </Row>

      <Modal
        open={showHelp}
        onCancel={() => setShowHelp(false)}
        footer={null}
        title="¿Cómo interpretar los diagnósticos de residuales?"
        width={650}
      >
        <h4>Serie Temporal de Residuales</h4>
        <ul style={{ lineHeight: 1.6 }}>
          <li><strong>Centro en cero:</strong> Los puntos deberían oscilar aleatoriamente alrededor de 0. Un sesgo sostenido indica error sistemático.</li>
          <li><strong>Varianza constante:</strong> El grosor de la banda de errores debe ser más o menos constante. Un "efecto embudo" (heterocedasticidad) sugiere que el error cambia con el tiempo.</li>
        </ul>

        <h4>Gráficos ACF y PACF</h4>
        <ul style={{ lineHeight: 1.6 }}>
            <li><strong>Objetivo:</strong> En un modelo bien ajustado, los errores son aleatorios y no deberían tener correlación entre sí.</li>
            <li><strong>Interpretación:</strong> Buscamos que todas las barras (después del retardo 0) estén **dentro** de la zona sombreada (banda de confianza).</li>
            <li><strong>Barras fuera de la banda:</strong> Si una o más barras sobresalen significativamente, indica que queda información predecible en los errores. Por ejemplo, una barra alta en el retardo 7 podría sugerir un patrón semanal que el modelo no capturó del todo.</li>
            <li><strong>Solución:</strong> Si ves patrones, considera añadir más `lags` en la configuración del modelo o crear nuevas características (features) que capturen esa dinámica.</li>
        </ul>
      </Modal>
    </Card>
  );
}
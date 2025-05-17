import React from 'react';
import { Line } from 'react-chartjs-2';
import { Card, Button, Modal } from 'antd';
import { QuestionCircleOutlined } from '@ant-design/icons';

export default function ResidualsDisplay({ dates, values }) {
  const [showHelp, setShowHelp] = React.useState(false);

  const data = {
    labels: dates,
    datasets: [{
      label: 'Residuos (y - ŷ)',
      data: values,
      fill: false,
      tension: 0,
      borderColor: '#ff4d4f',
      pointRadius: 2
    }]
  };
  const options = {
    scales: {
      x: { ticks: { maxRotation: 0 } },
      y: { title: { display: true, text: 'Error' } }
    },
    plugins: {
      legend: { display: false },
      title : { display: true, text: 'Serie temporal de residuales' }
    }
  };
  return (
    <Card style={{ position: 'relative' }}>
      {/* Botón “?” flotante */}
      <Button
        type="text"
        icon={<QuestionCircleOutlined />}
        onClick={() => setShowHelp(true)}
        style={{ position: 'absolute', top: 8, right: 8 }}
        aria-label="Ayuda sobre residuales"
      />

      <Line data={data} options={options} />

      {/* Modal de ayuda */}
      <Modal
        open={showHelp}
        onCancel={() => setShowHelp(false)}
        footer={null}
        title="¿Cómo interpretar los residuales?"
      >
        <ul style={{ lineHeight: 1.6 }}>
          <li><strong>Centro en cero:</strong> los puntos deberían oscilar alrededor de 0. Un sesgo sostenido indica error sistemático.</li>
          <li><strong>Ausencia de patrón:</strong> no debería verse tendencia ni ciclos regulares. Si aparecen, el modelo no capturó toda la dinámica temporal.</li>
          <li><strong>Varianza constante:</strong> cuidado con el “efecto embudo”; heterocedasticidad sugiere que el error crece o disminuye con el tiempo.</li>
          <li><strong>Valores atípicos:</strong> picos grandes pueden ser outliers o fallas de medición; vale la pena investigarlos.</li>
          <li><strong>Autocorrelación:</strong> aunque no se vea aquí, conviene revisar un correlograma (ACF) si sospechas correlación entre errores.</li>
        </ul>
      </Modal>
    </Card>
  );
}
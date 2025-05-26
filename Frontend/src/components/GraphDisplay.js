import React, { useRef } from 'react';
import { Line } from 'react-chartjs-2';
import { Button } from 'antd';
import { CameraOutlined, ReloadOutlined } from '@ant-design/icons';
import dayjs from 'dayjs';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale,
  Filler
} from 'chart.js';
import annotationPlugin from 'chartjs-plugin-annotation'; // Plugin de anotaciones
import 'chartjs-adapter-date-fns'; // Adaptador para fechas
import zoomPlugin from 'chartjs-plugin-zoom'; // Plugin de zoom y pan


ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale, // Registrar escala de tiempo
  zoomPlugin, // Registrar plugin de zoom
  annotationPlugin,
  Filler
);

function GraphDisplay({ historicalData, forecastData, ticker }) {
    const chartRef = useRef(null);

    const hasHistoricalData = historicalData?.dates?.length > 0 && historicalData?.values?.length > 0;
    const hasForecastData = forecastData?.length > 0;

    if (!hasHistoricalData && !hasForecastData) {
        return <p style={{ textAlign: 'center', padding: '20px' }}>Genera un pronóstico para ver la gráfica.</p>;
    }

    const historicalDates = hasHistoricalData ? historicalData.dates : [];
    const forecastDates = hasForecastData ? forecastData.map(p => p.date) : [];

    const allDates = [...new Set([...historicalDates, ...forecastDates])]
        .sort((a, b) => new Date(a) - new Date(b));

    const historicalMap = new Map(
        hasHistoricalData ? historicalData.dates.map((d, i) => [d, historicalData.values[i]]) : []
    );
    
    const forecastPredictionMap = new Map(
        hasForecastData ? forecastData.map(p => [p.date, p.prediction]) : []
    );
    const forecastLowerBoundMap = new Map(
        hasForecastData ? forecastData.map(p => [p.date, p.lower_bound]) : []
    );
    const forecastUpperBoundMap = new Map(
        hasForecastData ? forecastData.map(p => [p.date, p.upper_bound]) : []
    );

    const historicalMappedValues = allDates.map(date => historicalMap.get(date) ?? null);
    const forecastMappedPredictions = allDates.map(date => forecastPredictionMap.get(date) ?? null);
    const forecastMappedLowerBound = allDates.map(date => forecastLowerBoundMap.get(date) ?? null);
    const forecastMappedUpperBound = allDates.map(date => forecastUpperBoundMap.get(date) ?? null);

    const chartData = {
        labels: allDates,
        datasets: []
    };

    if (hasHistoricalData) {
         chartData.datasets.push({
            label: `Precio Histórico (${ticker})`,
            data: historicalMappedValues, 
            borderColor: 'rgb(75, 192, 192)',
            backgroundColor: 'rgba(75, 192, 192, 0.5)',
            tension: 0.1,
            pointRadius: 1, 
            pointHitRadius: 10,
            fill: false
        });
    }

    if (hasForecastData) {
        const hasCIData = forecastData.some(p => p.lower_bound !== undefined && p.upper_bound !== undefined);
        if (hasCIData) {
            chartData.datasets.push({
                label: 'Límite Inferior IC', 
                data: forecastMappedLowerBound,
                borderColor: 'transparent',
                borderWidth: 0, 
                pointRadius: 0,
                fill: false
            });

            chartData.datasets.push({
                label: 'Límite Superior IC', 
                data: forecastMappedUpperBound, 
                borderColor: 'transparent', 
                backgroundColor: 'rgba(255, 99, 132, 0.2)', 
                borderWidth: 1,
                pointRadius: 0,
                fill: '-1', 
            });
        }

        chartData.datasets.push({
            label: 'Pronóstico',
            data: forecastMappedPredictions, 
            borderColor: 'rgb(255, 99, 132)',
            backgroundColor: 'rgba(255, 99, 132, 0.5)',
            borderDash: [5, 5], // Línea punteada
            tension: 0.1,
            pointRadius: 2, 
            pointHitRadius: 10,
            fill: false
        });
    }

    const handleResetZoom = () => {
        if (chartRef.current) {
            chartRef.current.resetZoom();
        }
    }

    const handleExportImage = () => {
        if (chartRef.current) {
            const imageBase64 = chartRef.current.toBase64Image();
            const link = document.createElement('a');
            link.href = imageBase64;
            const filename = `grafico_${ticker}_${dayjs().format('YYYYMMDD_HHmmss')}.png`;
            link.download = filename;
            link.click();
        }
    };


    const options = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { position: 'top' },
            title: { display: true, text: `Serie de Tiempo y Pronóstico para ${ticker}` },
            zoom: { 
                pan: {
                    enabled: true, // Habilitar paneo (mover el gráfico)
                    mode: 'xy',   
                    threshold: 5, // Píxeles a mover antes de iniciar paneo
                },
                zoom: {
                    wheel: {
                        enabled: true, // Habilitar zoom con rueda del ratón
                    },
                    pinch: {
                        enabled: true // Habilitar zoom con "pellizco" en táctiles
                    },
                    mode: 'xy', 
                },
                limits: { 
                    x: {min: 'original', max: 'original'},
                    y: {min: 'original', max: 'original'}
                }
            } 
        },
        scales: {
        },
        interaction: {
            mode: 'index',
            intersect: false
        },
    };

    // Contenedor con altura definida es importante
    return (
        <div style={{ height: '450px', width: '100%', position: 'relative' }}>
            <Button
                icon={<ReloadOutlined />}
                onClick={handleResetZoom}
                style={{ position: 'absolute', top: '20px', right: '10px', zIndex: 10 }} 
                size="small"
                title="Restablecer Zoom" 
             />
            <Button
                icon={<CameraOutlined />}
                onClick={handleExportImage}
                style={{ position: 'absolute', top: '20px', right: '50px', zIndex: 10 }}
                size="small"
                title="Exportar Imagen"
            />
             <Line
                ref={chartRef} 
                options={options}
                data={chartData}
                key={ticker + (historicalData?.dates?.length || 0) + (forecastData?.length || 0)}
             />
        </div>
    );
    
}

export default GraphDisplay;
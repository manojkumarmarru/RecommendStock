import React, { useContext, useEffect, useRef, useState } from 'react';
import { Chart, registerables } from 'chart.js';
import zoomPlugin from 'chartjs-plugin-zoom';
import 'chartjs-adapter-date-fns';
import { TooltipContext } from './TooltipContext';
import '../Styles/StockChart.css'; // Import the CSS file

Chart.register(...registerables, zoomPlugin);

const timeRanges = {
    '1M': 30,
    '3M': 30 * 3,
    '6M': 30 * 6,
    '1Y': 365,
    '3Y': 365 * 3,
    '5Y': 365 * 5
};

const StockChart = ({ title, canvasId, datasets }) => {
    const chartRef = useRef(null);
    const { tooltipData, setTooltipData } = useContext(TooltipContext);
    const [selectedRange, setSelectedRange] = useState('6M');

    useEffect(() => {
        
        if (chartRef.current) {
            chartRef.current.destroy();
        }
        const ctx = document.getElementById(canvasId).getContext('2d');
        chartRef.current = new Chart(ctx, {
            type: 'line',
            data: {
                datasets: datasets
            },
            options: {
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'day',
                            tooltipFormat: 'PP',
                            displayFormats: {
                                day: 'MMM d, yyyy'
                            }
                        },
                        title: {
                            display: true,
                            text: 'Date'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Value'
                        }
                    }
                },
                plugins: {
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                if (context.parsed.y !== null) {
                                    label += context.parsed.y;
                                    if (context.dataset.label === 'Rating Score' && context.raw.recommendation) {
                                        label += ` (${context.raw.recommendation})`;
                                    }
                                }
                                return label;
                            },
                            afterBody: function(context) {
                                const date = context[0].label;
                                const tooltipInfo = context.map(item => ({
                                    label: item.dataset.label,
                                    value: item.raw.y
                                }));
                                setTooltipData({ date, tooltipInfo });
                            }
                        }
                    },
                    zoom: {
                        pan: {
                            enabled: true,
                            mode: 'x',
                        },
                        zoom: {
                            wheel: {
                                enabled: true,
                            },
                            pinch: {
                                enabled: true
                            },
                            mode: 'x',
                        }
                    }
                },
                hover: {
                    mode: 'index',
                    intersect: false
                }
            }
        });

        return () => {
            chartRef.current.destroy();
        };
    }, [canvasId, datasets, setTooltipData]);

    useEffect(() => {
        if (chartRef.current) {
            const allDates = datasets.flatMap(dataset => dataset.data.map(dataPoint => new Date(dataPoint.x)));
            const minDate = new Date(Math.min(...allDates));
            const maxDate = new Date(Math.max(...allDates));

            const now = new Date();
            const pastDate = new Date(now);
            pastDate.setDate(now.getDate() - timeRanges[selectedRange]);

            chartRef.current.options.scales.x.min = pastDate < minDate ? minDate : pastDate;
            chartRef.current.options.scales.x.max = maxDate;
            chartRef.current.update();
        }
    }, [datasets, selectedRange]);

    return (
        <div className="chart-container">
            <div className="chart-header">
                <h4>{title}</h4>
                <div className="time-range-buttons">
                    {Object.keys(timeRanges).map(range => (
                        <button
                            key={range}
                            onClick={() => setSelectedRange(range)}
                            className={selectedRange === range ? 'active' : ''}
                        >
                            {range}
                        </button>
                    ))}
                </div>
            </div>
            <canvas id={canvasId}></canvas>
        </div>
    );
};

export default StockChart;
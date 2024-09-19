import React, { useContext, useEffect, useRef } from 'react';
import { Chart, registerables } from 'chart.js';
import zoomPlugin from 'chartjs-plugin-zoom';
import 'chartjs-adapter-date-fns';
import { TooltipContext } from './TooltipContext';

Chart.register(...registerables, zoomPlugin);

const StockChart = ({ canvasId, datasets }) => {
    const chartRef = useRef(null);
    const { tooltipData, setTooltipData } = useContext(TooltipContext);

    useEffect(() => {
        const ctx = document.getElementById(canvasId).getContext('2d');

        if (chartRef.current) {
            chartRef.current.destroy();
        }

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
        if (chartRef.current && tooltipData) {
            const { date, tooltipInfo } = tooltipData;
            const dateIndex = chartRef.current.data.labels.findIndex(label => label === date);
            if (dateIndex !== -1) {
                const tooltip = chartRef.current.tooltip;
                tooltip.setActiveElements(
                    tooltipInfo.map((info, index) => ({
                        datasetIndex: index,
                        index: dateIndex
                    })),
                    { x: 0, y: 0 }
                );
                chartRef.current.update();
            }
        }
    }, [tooltipData]);

    return <canvas id={canvasId}></canvas>;
};

export default StockChart;
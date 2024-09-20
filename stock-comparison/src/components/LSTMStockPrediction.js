import React, { useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import 'chart.js/auto';
import '../Styles/LSTMStockPrediction.css';
import { addDays, format } from 'date-fns';

const LSTMStockPrediction = ({ symbol, predictionData }) => {
    const generateDates = (startDate, length) => {
        const dates = [];
        for (let i = 0; i < length; i++) {
            dates.push(format(addDays(new Date(startDate), i), 'yyyy-MM-dd'));
        }
        return dates;
    };

    const renderChart = () => {
        if (!predictionData || !predictionData.forecastedAdjClose || predictionData.forecastedAdjClose.length === 0) {
            return <div>No forecasted data available.</div>;
        }

        const startDate = '2023-01-01'; // Replace with the actual start date of the forecast
        const dates = generateDates(startDate, predictionData.forecastedAdjClose.length);

        const data = {
            labels: dates, // Use generated dates as labels
            datasets: [
                {
                    label: 'Forecasted Close',
                    data: predictionData.forecastedAdjClose, // Use forecasted prices as data
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    fill: false,
                },
            ],
        };

        const options = {
            responsive: true,
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'day',
                        tooltipFormat: 'P', // Correct format string
                    },
                    title: {
                        display: true,
                        text: 'Date',
                    },
                },
                y: {
                    beginAtZero: false,
                    title: {
                        display: true,
                        text: 'Stock Price',
                    },
                },
            },
        };

        return <Line data={data} options={options} />;
    };

    return (
        <div className="lstm-stock-prediction">
            <h1>LSTM Stock Prediction for {symbol}</h1>
            {renderChart()}
        </div>
    );
};

export default LSTMStockPrediction;
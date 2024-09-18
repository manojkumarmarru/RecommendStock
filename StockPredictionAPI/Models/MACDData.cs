using System;
namespace StockPredictionAPI.Models
{
    public class MACDData
    {
        public DateTime Date { get; set; }
        public float MACD { get; set; }
    }
}
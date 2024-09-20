using System;
namespace StockPredictionAPI.Models
{
    public class StockData
    {
        public DateTime Date { get; set; }
        public float Open { get; set; }
        public float High { get; set; }
        public float Low { get; set; }
        public float Close { get; set; }
        public float AdjClose { get; set; }
        public float Volume { get; set; }
        public float EMA { get; set; } 
        public float SMA { get; set; } 
        public float RSI { get; set; } 
        public float Sentiment { get; set; } 
    }
}
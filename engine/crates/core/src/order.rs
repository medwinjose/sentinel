use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Bid,
    Ask,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Order {
    pub id: u64,
    pub side: Side,
    pub price: i64,
    pub quantity: u64,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Trade {
    pub maker_order_id: u64,
    pub taker_order_id: u64,
    pub price: i64,
    pub quantity: u64,
    pub timestamp: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BookError {
    InvalidPrice,
    InvalidQuantity,
    DuplicateOrderId,
    OrderNotFound,
}

impl fmt::Display for BookError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BookError::InvalidPrice => write!(f, "Order price must be strictly positive"),
            BookError::InvalidQuantity => write!(f, "Order quantity must be strictly positive"),
            BookError::DuplicateOrderId => write!(f, "Order ID already exists in the book"),
            BookError::OrderNotFound => write!(f, "Order ID not found in the book"),
        }
    }
}

impl std::error::Error for BookError {}

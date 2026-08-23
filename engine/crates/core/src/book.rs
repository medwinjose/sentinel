use crate::order::{BookError, Order, Side, Trade};
use std::collections::{BTreeMap, HashMap, VecDeque};

#[derive(Debug, Default)]
pub struct OrderBook {
    /// Bids ordered by price ascending (best bid is max price, accessed via iter().rev())
    bids: BTreeMap<i64, VecDeque<Order>>,
    /// Asks ordered by price ascending (best ask is min price, accessed via iter())
    asks: BTreeMap<i64, VecDeque<Order>>,
    /// Fast lookup to support cancellation
    orders: HashMap<u64, Order>,
}

impl OrderBook {
    pub fn new() -> Self {
        Self {
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            orders: HashMap::new(),
        }
    }

    pub fn best_bid(&self) -> Option<i64> {
        self.bids.keys().next_back().copied()
    }

    pub fn best_ask(&self) -> Option<i64> {
        self.asks.keys().next().copied()
    }

    pub fn spread(&self) -> Option<i64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(b), Some(a)) => Some(a - b),
            _ => None,
        }
    }

    pub fn add_limit_order(&mut self, mut order: Order) -> Result<Vec<Trade>, BookError> {
        if order.price <= 0 {
            return Err(BookError::InvalidPrice);
        }
        if order.quantity == 0 {
            return Err(BookError::InvalidQuantity);
        }
        if self.orders.contains_key(&order.id) {
            return Err(BookError::DuplicateOrderId);
        }

        let mut trades = Vec::new();

        // Attempt to match against resting opposite side orders
        match order.side {
            Side::Bid => {
                let mut prices_to_remove = Vec::new();
                for (&ask_price, queue) in self.asks.iter_mut() {
                    if order.price < ask_price || order.quantity == 0 {
                        break;
                    }

                    while let Some(mut resting_ask) = queue.pop_front() {
                        let fill_qty = std::cmp::min(order.quantity, resting_ask.quantity);

                        trades.push(Trade {
                            maker_order_id: resting_ask.id,
                            taker_order_id: order.id,
                            price: ask_price,
                            quantity: fill_qty,
                            timestamp: order.timestamp,
                        });

                        order.quantity -= fill_qty;
                        resting_ask.quantity -= fill_qty;

                        // If resting order is partially filled, put it back
                        if resting_ask.quantity > 0 {
                            queue.push_front(resting_ask);
                            // Also update the order in the hashmap to reflect new quantity
                            self.orders.insert(resting_ask.id, resting_ask);
                        } else {
                            // Order is fully filled, remove from hashmap
                            self.orders.remove(&resting_ask.id);
                        }

                        if order.quantity == 0 {
                            break;
                        }
                    }

                    if queue.is_empty() {
                        prices_to_remove.push(ask_price);
                    }

                    if order.quantity == 0 {
                        break;
                    }
                }

                // Cleanup empty price levels
                for price in prices_to_remove {
                    self.asks.remove(&price);
                }

                // Rest remaining quantity
                if order.quantity > 0 {
                    self.orders.insert(order.id, order);
                    self.bids.entry(order.price).or_default().push_back(order);
                }
            }
            Side::Ask => {
                let mut prices_to_remove = Vec::new();
                for (&bid_price, queue) in self.bids.iter_mut().rev() {
                    if order.price > bid_price || order.quantity == 0 {
                        break;
                    }

                    while let Some(mut resting_bid) = queue.pop_front() {
                        let fill_qty = std::cmp::min(order.quantity, resting_bid.quantity);

                        trades.push(Trade {
                            maker_order_id: resting_bid.id,
                            taker_order_id: order.id,
                            price: bid_price,
                            quantity: fill_qty,
                            timestamp: order.timestamp,
                        });

                        order.quantity -= fill_qty;
                        resting_bid.quantity -= fill_qty;

                        if resting_bid.quantity > 0 {
                            queue.push_front(resting_bid);
                            self.orders.insert(resting_bid.id, resting_bid);
                        } else {
                            self.orders.remove(&resting_bid.id);
                        }

                        if order.quantity == 0 {
                            break;
                        }
                    }

                    if queue.is_empty() {
                        prices_to_remove.push(bid_price);
                    }

                    if order.quantity == 0 {
                        break;
                    }
                }

                for price in prices_to_remove {
                    self.bids.remove(&price);
                }

                if order.quantity > 0 {
                    self.orders.insert(order.id, order);
                    self.asks.entry(order.price).or_default().push_back(order);
                }
            }
        }

        Ok(trades)
    }

    pub fn cancel_order(&mut self, order_id: u64) -> Result<(), BookError> {
        if let Some(order) = self.orders.remove(&order_id) {
            match order.side {
                Side::Bid => {
                    if let Some(queue) = self.bids.get_mut(&order.price) {
                        queue.retain(|o| o.id != order_id);
                        if queue.is_empty() {
                            self.bids.remove(&order.price);
                        }
                    }
                }
                Side::Ask => {
                    if let Some(queue) = self.asks.get_mut(&order.price) {
                        queue.retain(|o| o.id != order_id);
                        if queue.is_empty() {
                            self.asks.remove(&order.price);
                        }
                    }
                }
            }
            Ok(())
        } else {
            Err(BookError::OrderNotFound)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_order(id: u64, side: Side, price: i64, quantity: u64) -> Order {
        Order {
            id,
            side,
            price,
            quantity,
            timestamp: 1000 + id, // Mock timestamp
        }
    }

    #[test]
    fn test_empty_book_insert() {
        let mut book = OrderBook::new();
        let order = create_order(1, Side::Bid, 100, 10);
        let trades = book.add_limit_order(order).unwrap();
        assert!(trades.is_empty());
        assert_eq!(book.best_bid(), Some(100));
        assert_eq!(book.best_ask(), None);
    }

    #[test]
    fn test_invalid_orders() {
        let mut book = OrderBook::new();
        let bad_price = create_order(1, Side::Bid, 0, 10);
        assert_eq!(
            book.add_limit_order(bad_price),
            Err(BookError::InvalidPrice)
        );

        let bad_qty = create_order(2, Side::Bid, 100, 0);
        assert_eq!(
            book.add_limit_order(bad_qty),
            Err(BookError::InvalidQuantity)
        );

        let order = create_order(3, Side::Bid, 100, 10);
        book.add_limit_order(order).unwrap();
        let dup = create_order(3, Side::Ask, 101, 10);
        assert_eq!(book.add_limit_order(dup), Err(BookError::DuplicateOrderId));
    }

    #[test]
    fn test_price_time_priority() {
        let mut book = OrderBook::new();

        // Add multiple resting asks at same price
        book.add_limit_order(create_order(1, Side::Ask, 100, 10))
            .unwrap();
        book.add_limit_order(create_order(2, Side::Ask, 100, 10))
            .unwrap();
        // Add ask at better (lower) price
        book.add_limit_order(create_order(3, Side::Ask, 99, 10))
            .unwrap();

        assert_eq!(book.best_ask(), Some(99));

        // Incoming bid crosses all of them
        let trades = book
            .add_limit_order(create_order(4, Side::Bid, 100, 30))
            .unwrap();

        assert_eq!(trades.len(), 3);
        // Should hit order 3 first (better price)
        assert_eq!(trades[0].maker_order_id, 3);
        assert_eq!(trades[0].price, 99);
        // Then order 1 (time priority over 2)
        assert_eq!(trades[1].maker_order_id, 1);
        assert_eq!(trades[1].price, 100);
        // Finally order 2
        assert_eq!(trades[2].maker_order_id, 2);
        assert_eq!(trades[2].price, 100);
    }

    #[test]
    fn test_partial_fill_and_rest() {
        let mut book = OrderBook::new();

        book.add_limit_order(create_order(1, Side::Ask, 100, 5))
            .unwrap();

        let trades = book
            .add_limit_order(create_order(2, Side::Bid, 100, 15))
            .unwrap();

        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].quantity, 5);

        // Remaining 10 should rest as bid
        assert_eq!(book.best_bid(), Some(100));
        assert_eq!(book.best_ask(), None);

        // Verify resting order details
        let resting = book.orders.get(&2).unwrap();
        assert_eq!(resting.quantity, 10);
    }

    #[test]
    fn test_cancel_order() {
        let mut book = OrderBook::new();

        // Cancel non-existent
        assert_eq!(book.cancel_order(999), Err(BookError::OrderNotFound));

        // Cancel existing
        book.add_limit_order(create_order(1, Side::Bid, 100, 10))
            .unwrap();
        assert_eq!(book.best_bid(), Some(100));

        book.cancel_order(1).unwrap();
        assert_eq!(book.best_bid(), None);
        assert!(!book.orders.contains_key(&1));

        // Verify it doesn't match after cancellation
        let trades = book
            .add_limit_order(create_order(2, Side::Ask, 100, 10))
            .unwrap();
        assert!(trades.is_empty());
    }
    #[test]
    fn test_full_fill_exact_match() {
        let mut book = OrderBook::new();
        book.add_limit_order(create_order(1, Side::Ask, 100, 10))
            .unwrap();

        let trades = book
            .add_limit_order(create_order(2, Side::Bid, 100, 10))
            .unwrap();

        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].quantity, 10);
        assert_eq!(book.best_bid(), None);
        assert_eq!(book.best_ask(), None);
        assert!(!book.orders.contains_key(&1));
        assert!(!book.orders.contains_key(&2));
    }

    #[test]
    fn test_partial_fill_across_multiple_levels() {
        let mut book = OrderBook::new();
        book.add_limit_order(create_order(1, Side::Ask, 100, 10))
            .unwrap();
        book.add_limit_order(create_order(2, Side::Ask, 101, 10))
            .unwrap();

        // Incoming bid of 15 crosses 100 and partially fills 101
        let trades = book
            .add_limit_order(create_order(3, Side::Bid, 101, 15))
            .unwrap();

        assert_eq!(trades.len(), 2);
        assert_eq!(trades[0].price, 100);
        assert_eq!(trades[0].quantity, 10);
        assert_eq!(trades[1].price, 101);
        assert_eq!(trades[1].quantity, 5);

        // Remainder of order 2 should still rest
        assert_eq!(book.best_ask(), Some(101));
        let resting = book.orders.get(&2).unwrap();
        assert_eq!(resting.quantity, 5);
    }
}

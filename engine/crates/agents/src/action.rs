use sentinel_core::order::Order;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentAction {
    PlaceOrder(Order),
    CancelOrder { timestamp: u64, order_id: u64 },
}

impl AgentAction {
    pub fn timestamp(&self) -> u64 {
        match self {
            AgentAction::PlaceOrder(o) => o.timestamp,
            AgentAction::CancelOrder { timestamp, .. } => *timestamp,
        }
    }
}

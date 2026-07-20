def calculate_order_size(account_balance: float, current_oil_price: float, risk_pct: float = 0.01):
    """
    Calculates exactly how many units of Oil to buy based on a strict 1% risk rule.
    """
    # 1. How much money are we willing to risk? (1% of £100,000 = £1,000)
    capital_at_risk = account_balance * risk_pct
    
    # 2. Assume our Stop Loss is $2.00 away from the current price
    stop_loss_distance = 2.00 
    
    # 3. Calculate position size: Risk / Stop Distance
    # If we risk £1000, and our stop is $2 away, we can buy 500 barrels.
    units_to_buy = capital_at_risk / stop_loss_distance
    
    print(f"\n🛡️ RISK MANAGER SHIELD ACTIVE")
    print(f"Account Balance: £{account_balance}")
    print(f"Max Allowed Risk: £{capital_at_risk}")
    print(f"Approved Trade Size: {round(units_to_buy, 2)} units of WTI")
    
    return round(units_to_buy, 2)

if __name__ == "__main__":
    # Test it with your OANDA Demo Balance
    demo_balance = 100000.00
    current_wti_price = 82.50
    calculate_order_size(demo_balance, current_wti_price)
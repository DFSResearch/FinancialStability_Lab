###class containing different methods for fast way to approximate ytm
class approx_ytm():
    '''
    Several techiques for rough approximation of ytm
    
    Parameters
    --------------
    data: Pandas Series
        Series is a row of bonds' DataFrame
    price: float
        Standartized price of deal
    '''
    def __init__(self, data, price_hat):
        self.coupon = data['coupon_rate'] * 100 #100 - standartized face value of bonds
        self.price = price_hat
        self.span = data['span']
        self.n_years = self.span / 365 #span given in days; need to be converterd in years format
        self.ann_amort = (100 - self.price) / self.n_years
        self.is_discount = data['bond_symb'] in ['NTK', 'MKM'] 
        
    #ytm for discount bonds -- e.g. NTK, MKM bonds    
    def discount_ytm(self):
        ytm = (100 - self.price) / self.span * 365 / 100
        return ytm
    
    def traditional(self):
        #checking if it is discount
        if self.is_discount:
            ytm = self.discount_ytm()
        else:
            ytm = (self.coupon + self.ann_amort) / (100 + self.price) / 2
        return ytm
    
    def henderson(self):
        #checking if it is discount
        if self.is_discount:
            ytm = self.discount_ytm()
        else:
            ytm = ((self.coupon + self.ann_amort) / 
                   (100 + 0.6 * (self.price - 100)))
        return ytm
    
    def sp500(self):
        #checking if it is discount
        if self.is_discount:
            ytm = self.discount_ytm()
        else:
            ytm = (0.5 * (self.coupon + self.ann_amort) * 
               (1 / self.price + 1 / (100 - self.ann_amort)))
        return ytm
    
    def griffith(self):
        #checking if it is discount
        if self.is_discount:
            ytm = self.discount_ytm()
        else:
            ytm = (0.5 * (self.coupon + self.ann_amort) * 
               (1 / self.price + 1 / (100 - self.ann_amort - self.n_years)))
        return ytm
    
    def todhunter(self):
        #checking if it is discount
        if self.is_discount:
            ytm = self.discount_ytm()
        else:
            ytm = ((self.coupon + self.ann_amort) / 
               (100 + (self.n_years + 1) / (self.n_years * (self.price - 100)))
              )
        return ytm
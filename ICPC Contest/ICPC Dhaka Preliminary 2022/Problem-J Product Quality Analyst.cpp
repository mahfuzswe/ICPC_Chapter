#include<bits/stdc++.h>
using namespace std;

typedef long long ll;

int main()
{
    ios_base::sync_with_stdio(false);
    cin.tie(0);
    
    int t; cin>>t;
    
    for(int i=1; i<=t; i++){
        
        int n; cin>>n;
        ll k; cin>>k;

        ll arr[n];

        ll f_sum = 0;
        ll max = 0;

        for(int i=0; i<n; i++){
            cin>>arr[i];

            f_sum += arr[i];

            if (arr[i] > max) {
            max = arr[i];
            }
        }

        for(int i = 0; i<k-1; i++){
            f_sum += max;
        }
        cout<<"Case "<<i<<": "<<f_sum<<endl;
    }

    return 0;
}


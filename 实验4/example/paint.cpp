#include <iostream>
using namespace std;

void sort(pair<int, int> list[], int n) {
	for (int i = 0; i < n; i++) {
		for (int j = i; j < n; j++) {
			if (list[i].first > list[j].first) {
				pair<int, int> temp = list[i];
				list[i] = list[j];
				list[j] = temp;
			}
		}
	}
	for (int i = 0; i < n; i++) {
		for (int j = i; j < n; j++) {
			if (list[i].first == list[j].first && list[i].second > list[j].second) {
				pair<int, int> temp = list[i];
				list[i] = list[j];
				list[j] = temp;
			}
		}
	}
}

int caculate(pair<int, int> list_red[], pair<int, int> list_blue[], int m, int n){
	int point = list_blue[0].second;
	int count = 0;
	int i = 0;
	int j = 0;
	while (true) {
		while(list_red[i].second < list_blue[j].first && i <= m - 1){
			i ++;
		}
		if(list_red[i].first > point || i >= m){
			count = -1;
			break;
		}
		while(list_red[i].first <= point && i <= m - 1){
			i++;
		}
		i --;
		while(list_blue[j].first <= list_red[i].second && j <= n - 1){
			j++;
		}
		point = list_blue[j].second;
		count++;
		if(j - 1 == n - 1) break;
	}
	return count;
}

int main() {
	int n;
	cin >> n;
	pair<int, int> list_blue[100000];
	for (int i = 0; i < n; i++) {
		cin >> list_blue[i].first >> list_blue[i].second;
	}
	int m;
	cin >> m;
	pair<int, int> list_red[100000];
	for (int i = 0; i < m; i++) {
		cin >> list_red[i].first >> list_red[i].second;
	}
	sort(list_blue, n);
	sort(list_red, m);
	int count = caculate(list_red, list_blue, m, n);
	cout << count << endl;

}
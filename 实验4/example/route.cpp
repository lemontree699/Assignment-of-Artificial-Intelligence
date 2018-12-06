#include <iostream>
#include <stack>
using namespace std;

#define size 1000

int min(int x, int y){
	return x < y ? x : y; 
}

int dynamic(int x, int y, int map[size][size], int len[size][size]){
	if(len[x][y] != 0) return len[x][y];
	if(x == 0 && y == 0){
		return map[0][0];
	}
	else if(x == 0){
		len[x][y] = dynamic(x, y - 1, map, len) + map[x][y];
		return len[x][y];
	
	}
	else if(y == 0) {
		len[x][y] = dynamic(x - 1, y, map, len) + map[x][y];
		return len[x][y];
	}
	else {
		int len1 = dynamic(x - 1, y, map, len) + map[x][y];
		int len2 = dynamic(x, y - 1, map, len) + map[x][y];
		if(len1 < len2){
			len[x][y] = len1;
			return len1;
		}
		else{
			len[x][y] = len2;
			return len2;
		}
	}
}

int main(){
	int M, N;
	cin >> N >> M;
	int map[size][size];
	int len[size][size];
	stack<char> route;
	for(int i = 0; i < M; i ++){
		for(int j = 0; j < N; j ++){
			cin >> map[i][j];
			len[i][j] = 0;
		}
	}
	int length = dynamic(M - 1, N - 1, map, len);
	int i = M - 1, j = N - 1;
	while(i != 0 || j != 0){
		if(j != 0 && (i == 0 || len[i][j - 1] <= len[i - 1][j])){
			route.push('R');
			j --;
		}
		if(i != 0 && (j == 0 || len[i][j - 1] > len[i - 1][j])){
			route.push('D');
			i --;
		}
	}
	while(!route.empty()){
		cout << route.top();
		route.pop();
	}
	cout << endl;
}
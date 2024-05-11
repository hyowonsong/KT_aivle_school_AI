// Javascript
// 웹브라우져에서 이벤트를 처리하는 문법
/*
1. 변수선언 : RAM 사용하는 문법
2. 데이터타입 : RAM 효율적으로 사용하는 문법 : 동적타이핑
3. 연산자 : CPU 사용하는 문법 : 산술, 비교, 논리
4. 조건문 : 조건에 따라서 다른 코드를 실행
5. 반복문 : 특정 코드를 반복적으로 실행
6. 함수 : 중복코드를 묶어서 코드 작성 실행
7. 객체 : 식별자 1개에 데이터 여러개 저장 문법 : 클래스

특징 : 인터프리터언어, 동적타이핑, 객체지향
*/

// 1. 변수선언
// 식별자 : 저장공간을 구별하는 문자열
// 식별자규칙 : 대소문자, 숫자, _, $ : 가장 앞에 숫자 X : 예약어 사용 X
// 식별자컨벤션 : camelCase(변수,함수), PascalCase(모듈)

// 식별자 1개, 데이터 1개
var data1 = 10;

// 식별자 n개, 데이터 n개
var data2 = 20, data3 = 30;
var data2 = 20,
    data3 = 30;

// 식별자 n개, 데이터 1개
var data4 = data5 = 40;

console.log(data1, data2, data3, data4, data5);

// 식별자 1개, 데이터 n개 : array

// 2. 데이터타입
// number, string, boolean, function, object
// 동적타이핑 : 데이터타입 선언 없이 자동으로 데이터타입 정의
let data1 = 1;
const data2 = 'js';
var data3 = true;
var data4 = function(){ console.log('js'); };
var data5 = {key: 'js'};

console.log(typeof data1, typeof data2, typeof data3);
console.log(typeof data4, typeof data5);

// 데이터가 없음
// undefined : 선언은 되었으나 데이터가 할당 X
// null : 선언은 되었으나 데이터 없음이 할당
// NaN : Number 데이터 타입에서 undefined
var data1 = undefined;
var data2 = null;
var data3 = NaN;
console.log(typeof data1, data1);
console.log(typeof data2, data2);
console.log(typeof data3, data3);

console.log(NaN == NaN, NaN > NaN);

// 데이터 타입의 형변환
// Number(), String(), Boolean()
var data1 = '1';
var data2 = 2;
console.log(typeof data1, typeof Number(data1));
console.log(data1 + data2);
console.log(Number(data1) + data2);
console.log(data1 + String(data2));

// 묵시적 형변환 : 서로 다른 데이터타입을 연산할때 데이터타입을 변경
var data1 = '1';
var data2 = 2;
console.log(typeof data1, typeof (data1 - 0));
console.log(typeof data2, typeof ('' + data2));

// 3. 연산자 : operator
// 산술 : 데이터 + 데이터 = 데이터 : +, -, *, /, %, **, ++, --
// 비교 : 데이터 + 데이터 = 논리값 : 조건 1개 : ==, !=, >, <, >=, <=, ===, !==
// 논리 : 논리값(조건1) + 논리값(조건2) = 논리값 : 조건 2개 이상 : !(not), &&(and), ||(or)
// 할당 : 변수 <산술>= 데이터 : 누적해서 연산
// 삼항 : true if condition else false (python) : condition ? ture : false (javascript)

// 산술
var data1 = 11, data2 = 4;
console.log(data1 / data2, data1 % data2, data2 ** 3);
// ++data1 : +1 하고 데이터 대입
var data1 = 5, data2 = 7;
data1 = ++data2;
console.log(data1, data2);
// data1++ : 데이터 대입 하고 +1
var data1 = 5, data2 = 7;
data1 = data2++;
console.log(data1, data2);

// 비교연산자 : ==(데이터만비교), ===(데이터,데이터타입비교)
var data1 = 1, data2 = '1';
console.log(data1, data2);
console.log(data1 == data2, data1 === data2);

// 논리연산자 : !, &&(and: T and T = T), ||(or: F or F = F)
console.log(true && false, true || false);

// 비교 : 조건 1개
// 논리 : 조건 2개 이상
// 예금잔고에서 예금인출이 가능하면 true, 아니면 false
// 조건 1 : 예금잔고 >= 인출금액
var balance = 10000;
var amount = 6000;
console.log(balance >= amount);

// 조건 2 : 최대 인출금액 5000원
var balance = 10000;
var amount = 6000;
console.log(balance >= amount, amount <= 5000);
console.log(balance >= amount && amount <= 5000);

// 할당연산자
var data1 = 10;
data1 += 20;
console.log(data1);

// 삼항연산자 : 조건 ? 참 : 거짓
var balance = 10000, amount = 16000;
var msg = balance >= amount ? '인출가능' : '인출불가';
console.log(msg);

// 실수하기 쉬운 코드 : 부동 소수점 에러
var data1 = 0.1, data2 = 0.2;
console.log(data1 + data2 === 0.3);
console.log(data1 + data2);
console.log(Math.round((data1 + data2) * 10) / 10);
console.log(Math.round((data1 + data2) * 10) / 10 === 0.3);

// 4. 조건문
// if, else if, else
var balance = 3000, amount = 4000;
if(balance < amount){
    console.log('인출불가:잔액부족');
} else if(amount > 5000){
    console.log('인출불가:최대인출금액초과');
} else {
    console.log('인출가능');
}

// 5. 반복문
// while, for, break, continue
// while : 횟수를 특정할수 없는 경우, for : 횟수를 특정할수 있는 경우
var count = 3;
while(count > 0){
    count -= 1;
    console.log('js');
}
// continue : 실행되면 반복문을 정의하는 코드로 올라가서 코드 실행
// break : 반복중단
for(var i = 0; i < 5; i++){
    if(i === 2){
        continue;
    }
    console.log('js', i);
    if(i >= 3){
        break;
    }
}

// 로또 번호 출력 : 1 ~ 45 랜덤한 숫자 6개 출력(문자열)
// 중복 허용 출력
var count = 6, lotto = '';
console.log(Math.random());
console.log(Math.ceil(4.2));
for(var i = 0; i < count; i++){
    var rnum = Math.ceil(Math.random() * 44) + 1;
    // var rnum = Math.floor(Math.random() * 45) + 1;
    lotto = lotto + rnum + ' ';
}
console.log(lotto);

// 6. 함수 : function
// 중복되는 코드를 묶어서 코드 작성 실행 문법 > 코드 유지 보수 향상
// 사용법 : 함수선언(코드작성) > 함수호출(코드실행)

// 로또번호 출력
var count = 6, lotto = '';
for(var i = 0; i < count; i++){
    var rnum = Math.floor(Math.random() * 50) + 1;
    lotto = lotto + rnum + ' ';
}
console.log(lotto);

// javascript 문자열 출력
console.log('javascript');

// 로또번호 출력
var count = 6, lotto = '';
for(var i = 0; i < count; i++){
    var rnum = Math.floor(Math.random() * 50) + 1;
    lotto = lotto + rnum + ' ';
}
console.log(lotto);

// 함수선언 : 코드작성
// parameter, argument : 함수 호출하는 코드에서 함수 선언하는 코드로 데이터 전달
function showLotto(count){
    var lotto = '';
    for(var i = 0; i < count; i++){
        var rnum = Math.floor(Math.random() * 50) + 1;
        lotto = lotto + rnum + ' ';
    }
    console.log(lotto);
}
// 로또번호출력 7개 : 함수호출 : 코드실행
showLotto(7);
// javascript출력
console.log('javascript');
// 로또번호출력 6개
showLotto(6);

// 1. 변수선언 : 식별자 규칙($), 컨벤션(camelCase,PascalCase)
// 2. 데이터타입 : number, string, boolean, object, function
// 3. 연산자 : 산술, 비교, 논리, 할당, 삼항
// 4. 조건문 : if, else if, else
// 5. 반복문 : while, for(var i=0; i<10; i++){}, break, continue
// 6. 함수 : 중복코드를 묶음 > 코드 유지보수 향상 : 함수선언 > 함수호출 : parameter, argument

function showLotto(count){
    var lotto = '';
    for(var i = 0; i < count; i++){
        var rnum = Math.floor(Math.random() * 50) + 1;
        lotto = lotto + rnum + ' ';
    }
    console.log(lotto);
}
console.log(typeof showLotto);

// 함수선언방법 1 : 선언식
function plus1(n1, n2){
    console.log(n1 + n2);
}
plus1(1, 2);

// 함수선언방법 2 : 표현식
var plus2 = function(n1, n2){
    console.log(n1 + n2);
}
plus2(2, 3);

// 표현식은 함수를 선언해야 함수 호출이 가능
plus2(2, 3);
var plus2 = function(n1, n2){
    console.log(n1 + n2);
}

// 선언식은 함수를 호출하고 선언해서 사용 가능 : 호이스팅
plus1(1, 2);
function plus1(n1, n2){
    console.log(n1 + n2);
}

// 스코프
// 함수안 : 지역영역 : 지역변수 : local
// 함수밖 : 전역영역 : 전역변수 : global

var data = 10;
function change(){
    data = 20; // var 사용하지 않으면 global 변수
}
change();
console.log(data);

var data = 10;
function change(){
    var data = 20; // var 사용하면 local 변수
}
change();
console.log(data);

// return
function plus1(n1, n2){
    console.log('plus1', n1 + n2);
    return n1 + n2;
}
function plus2(n1, n2){
    console.log('plus2', n1 + n2);
}
result1 = plus1(2, 3);
result2 = plus2(2, 3);
console.log(result1, result2);

// 익명함수 : 선언과 동시에 호출하는 함수
// 자바스크립트 코드를 웹서비스 사용자가 사용할수 없도록 하기 위해 사용
function plus1(n1, n2){
    console.log(n1 + n2);
}
plus1(2, 3);

(function plus1(n1, n2){
    console.log(n1 + n2);
}(2, 3))
plus1(2, 3);

// default parameter 설정
var plus = function(n1, n2){
    console.log(n1, n2);
    n2 = n2 || 10;
    return n1 + n2;
}
result = plus(1, 2)
console.log(result);

// 7. 객체 : object
// Array : 배열
var data = [1, 2, 3, 'A', 'B'];
console.log(typeof data, data);
// 데이터추가 : push()
data.push('C');
console.log(data);
console.log(data[3]);
for(var i=0; i < data.length; i++){
    console.log(i, data[i])
}

// 객체 생성 1
var data = {name: 'andy', plus: function(n1, n2){return n1 + n2} };
console.log(typeof data, data);
console.log(data.name, data['name']);

// 객체 생성 2
function Person(name){
    this.name = name;
}
var person = new Person('andy');
console.log(typeof person, person);

// 변수 추가
var data = {};
data.name = 'alice';
data['age'] = 23;
delete data.age;
console.log(data);

// json object : 웹에서 데이터를 주고 받는 용도로 사용되는 데이터 포멧
// 웹에서는 객체 자체사용이 불가능
// json object > str
var data = {name: 'peter', age: 29, skills: ['code', 'read']};
var json_str = JSON.stringify(data);
console.log(typeof data, typeof json_str);
console.log(data, json_str);
// str > json object
var json_obj = JSON.parse(json_str);
console.log(typeof json_obj, json_obj, json_obj.skills);

// 웹브라우져 객체
// window   : 전역객체 : 모든 변수와 함수와 객체를 저장하는 객체
// location : URL 데이터를 저장하는 객체
// document : 페이지 문서(HTML)에 대한 데이터를 저장하는 객체
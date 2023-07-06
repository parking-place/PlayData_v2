# 데이터베이스 목록 보기
SHOW databases;

# 데이터베이스 생성
CREATE DATABASE IF NOT EXISTS testdb;

# 데이터 베이스 사용
USE testdb;

# 데이터베이스 삭제
-- DROP DATABASE IF EXISTS testdb;

# 테이블 생성
CREATE TABLE IF NOT EXISTS mytable (            
    id INT UNSIGNED NOT NULL AUTO_INCREMENT,-- UNSIGNED : 양수만 허용, AUTO_INCREMENT : 자동 증가
    name VARCHAR(50) NOT NULL,              -- NOT NULL : NULL 허용 안함
    modelnumber VARCHAR(15) NOT NULL,       -- VARCHAR(15) : 최대 15자 char
    series VARCHAR(30) NULL,                -- NULL : NULL 허용
    PRIMARY KEY(id)                         -- PRIMARY KEY : 기본키
);

# 새 테이블 생성
CREATE TABLE newtable (
    id INT UNSIGNED NOT NULL AUTO_INCREMENT,
    name CHAR(20) NOT NULL,
    age TINYINT,                -- TINYINT : -128 ~ 127 (1byte)
    phone VARCHAR(20),
    email VARCHAR(30) NOT NULL,
    address VARCHAR(50),
    PRIMARY KEY(id)
);

# 테이블 목록 보기
SHOW TABLES;

# 테이블의 구조 보기 (Descoption)
DESC mytable;

DESC newtable;

# 새로운 컬럼 추가 
ALTER TABLE mytable ADD COLUMN new_column varchar(10) NOT NULL;
# 결과 확인 
desc mytable;

# 컬럼 타입 변경 
ALTER TABLE mytable MODIFY COLUMN modelnumber varchar(20) NOT NULL;
# 결과 확인 
desc mytable;

# 컬럼 이름 변경 
ALTER TABLE mytable CHANGE COLUMN modelnumber new_modelnumber varchar(10) NOT NULL;
# 결과 확인 
desc mytable;

# 컬럼 삭제 
ALTER TABLE mytable DROP COLUMN series;
# 결과 확인 
desc mytable;

# 테이블 삭제 
DROP TABLE IF EXISTS mytable;
# 결과 확인 
show tables;
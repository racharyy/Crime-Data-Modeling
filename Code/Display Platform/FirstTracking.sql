select * from tracks;
select * from track_points;
select * from track_points_ext_video;
select * from missions;

show columns from track_points;


select LOWER(CONCAT(
    SUBSTR(HEX(track_uuid), 1, 8), '-',
    SUBSTR(HEX(track_uuid), 9, 4), '-',
    SUBSTR(HEX(track_uuid), 13, 4), '-',
    SUBSTR(HEX(track_uuid), 17, 4), '-',
    SUBSTR(HEX(track_uuid), 21)
  )) as TRACKID , target_latitude, target_longitude, timestamp_epoch_us from track_points order by TRACKID, timestamp_epoch_us;
 
 

SELECT
  LOWER(CONCAT(
    SUBSTR(HEX(track_uuid), 1, 8), '-',
    SUBSTR(HEX(track_uuid), 9, 4), '-',
    SUBSTR(HEX(track_uuid), 13, 4), '-',
    SUBSTR(HEX(track_uuid), 17, 4), '-',
    SUBSTR(HEX(track_uuid), 21)
  ))
FROM track_points;


select(HEX(track_uuid)) as TRACKID , target_latitude, target_longitude, timestamp_epoch_us from track_points order by TRACKID, timestamp_epoch_us;

select distinct HEX(track_uuid) as TRACKID from track_points;


show tables from mysql;


select(HEX(track_uuid)) as TRACKID , target_latitude, target_longitude, timestamp_epoch_us from track_points order by TRACKID, timestamp_epoch_us;


-- gets the endpoint
select distinct HEX(track_uuid) as TRACKID from track_points;

select HEX(track_uuid) as TRACKID , target_latitude, target_longitude, timestamp_epoch_us 
from track_points where HEX(track_uuid) =  '000004789C2F4EAD821E149A5D7DAB51'   order by timestamp_epoch_us desc limit 1;






use TASS_ElmGrove;

use TASS_Iron;
select  track_uuid , target_latitude, target_longitude, timestamp_epoch_us from track_points  where track_uuid  in  
(select distinct track_uuid  from track_points) order by timestamp_epoch_us desc limit 1;


select(HEX(track_uuid)) as TRACKID 
								, target_latitude
                                , target_longitude
                                , timestamp_epoch_us 
                                 from track_points order by TRACKID, timestamp_epoch_us limit 200;


-- experimenting with stored procedures

CREATE PROCEDURE TEST()
select distinct HEX(track_uuid) as TRACKID from track_points;

CALL TEST();

DROP PROCEDURE IF EXISTS ENDPOINT;
DELIMITER //
 
CREATE PROCEDURE `p2` ()
LANGUAGE SQL
COMMENT 'A procedure'
BEGIN
    SELECT 'Hello World !';
END//

DELIMITER ; 
call p2();

--- STORED Procedure for ENDPOINT

DROP PROCEDURE IF EXISTS ENDPOINT;
DELIMITER //
 
CREATE PROCEDURE `ENDPOINT` ()
LANGUAGE SQL
COMMENT 'A procedure'
BEGIN
    DECLARE finished INT default 0;
    DECLARE TRK varchar(32)  CHARACTER SET utf8 DEFAULT "";
    DECLARE cur CURSOR FOR   select distinct HEX(track_uuid) as TRACKID from track_points;
    DECLARE CONTINUE HANDLER FOR NOT FOUND SET finished = 1;
    OPEN cur;
    get_tracks: LOOP
    
	 FETCH cur into TRK;
     
     IF finished = 1 THEN
	 LEAVE get_tracks;
     END IF;
     INSERT into ENDPOINTS ( track_id,end_latitude,end_longitude, end_time)
      Select HEX(track_uuid) as TRACKID, target_latitude,target_longitude,  timestamp_epoch_us from track_points where HEX(track_uuid) in (TRK) order by timestamp_epoch_us desc limit 1;
     END LOOP get_tracks;
    close cur;
    
END//

DELIMITER ; 



-- use TASS_Iron;
use TASS_Elmgrove;

DROP TABLE ENDPOINTS;

CREATE TABLE ENDPOINTS 
(

  track_id varchar(32) CHARACTER SET utf8,
  end_latitude double,
  end_longitude double,
  end_time bigint(20),
  CONSTRAINT pk_track_id PRIMARY KEY (track_id)
);
DESC tass_elmgrove.ENDPOINTS;


call ENDPOINT();

select * from ENDPOINTS;



-- not working... need to insert into a table in the stored procedure
select HEX(track_uuid) as TRACKID , target_latitude, target_longitude, timestamp_epoch_us 
from track_points where HEX(track_uuid) in (select  @Results)   order by timestamp_epoch_us desc limit 1;


select distinct HEX(track_uuid) as TRACKID from track_points;



DROP PROCEDURE IF EXISTS ENDPOINT;




-- work on the query to seledt start and end 

select(HEX(track_uuid)) as TRACKID 
								, target_latitude as start_latitude
                                , target_longitude as start_longitude
                                , timestamp_epoch_us as start_time
                                 from track_points where HEX(track_uuid) in ( select distinct  HEX(track_uuid) from track_points) order by timestamp_epoch_us desc limit 1;
         
         
select HEX(track_uuid)  as TRACKID 
								, target_latitude 
                                , target_longitude 
                                , timestamp_epoch_us 
                                 from track_points where HEX(track_uuid) in ( '000004789C2F4EAD821E149A5D7DAB51'
                                 , '000032E906424F4481304A535D1F89E3')       order by HEX(track_uuid),  timestamp_epoch_us ;
                                 
-- describe track_points
DESC track_points;
DROP TABLE ENDPOINTS;

CREATE TABLE ENDPOINTS 
(

  track_id varchar(32) CHARACTER SET utf8,
  end_latitude double,
  end_longitude double,
  end_time bigint(20),
  CONSTRAINT pk_track_id PRIMARY KEY (track_id)
);



 
INSERT into ENDPOINTS (
track_id
,end_latitude
,end_longitude
, end_time)
select(HEX(track_uuid)) as TRACKID 
								, target_latitude 
                                , target_longitude 
                                , timestamp_epoch_us 
                                 from track_points 
                                 where HEX(track_uuid) =   '000004789C2F4EAD821E149A5D7DAB51'  order by timestamp_epoch_us desc limit 1;

                                 
                                 
                                 
select * from ENDPOINTS;



